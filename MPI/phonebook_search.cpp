//... To compile: mpic++ phonebook_search.cpp -o phonebook_search
//... To run: mpirun -n 4 ./phonebook_search phonebook1.txt phonebook2.txt

#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

void sendString(string text, int receiver)
{
    int length = text.size() + 1;

    // Send the length of the string to the specified receiver process
    MPI_Send(&length, 1, MPI_INT, receiver, 1, MPI_COMM_WORLD);

    // Send the actual string data to the specified receiver process
    MPI_Send(&text[0], length, MPI_CHAR, receiver, 1, MPI_COMM_WORLD);
}

string receiveString(int sender)
{
    int length;

    // Receive the length of the incoming string from the specified sender process
    MPI_Recv(&length, 1, MPI_INT, sender, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    char *text = new char[length];

    // Receive the actual string data from the specified sender process
    MPI_Recv(text, length, MPI_CHAR, sender, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    return string(text);
}

string vectorToString(vector<string> &names, int start, int end)
{
    string text = "";
    end = min(end, int(names.size()));

    for (int i = start; i < end; ++i)
    {
        text += names[i] + "\n";
    }
    return text;
}

vector<string> stringToVector(string text)
{
    stringstream temp(text); // Create a stringstream to tokenize(break it down into individual pieces) the input string

    vector<string> names;
    string name;

    while (temp >> name) // Tokenize the input string and store individual strings in the vector
    {
        names.push_back(name);
    }

    return names;
}

void readPhoneBook(vector<string> &fileNames, vector<string> &names, vector<string> &phoneNumbers)
{
    for (auto fileName : fileNames)
    {
        ifstream file(fileName); // Open the file specified by the current fileName
        string name, number;

        while (file >> name >> number) // Read names and phone numbers from the file until the end of the file is reached
        {
            names.push_back(name);
            phoneNumbers.push_back(number);
        }

        file.close(); // Close the file after reading all the data
    }
}

void check(string name, string phone, string searchName, int rank)
{
    if (name.size() != searchName.size())
    {
        return;
    }
    for (int i = 0; i < searchName.size(); i++)
    {
        if (name[i] != searchName[i])
        {
            return;
        }
    }
    printf("%s %s found by process %d.\n", name.c_str(), phone.c_str(), rank);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    double startTime = MPI_Wtime();

    string searchName;

    if (rank == 0)
    {
        vector<string> names, phoneNumbers;
        vector<string> fileNames(argv + 1, argv + argc);

        readPhoneBook(fileNames, names, phoneNumbers);

        int segments = 1 + names.size() / worldSize;

        cout << "Enter a name to search: ";
        cin >> searchName;

        for (int i = 1; i < worldSize; ++i)
        {
            int start = i * segments;
            int end = start + segments;
            string namesToSend = vectorToString(names, start, end);
            string phoneNumbersToSend = vectorToString(phoneNumbers, start, end);

            sendString(namesToSend, i);
            sendString(phoneNumbersToSend, i);
            sendString(searchName, i);
        }

        for (int i = 0; i < segments; i++)
        {
            check(names[i], phoneNumbers[i], searchName, rank);
        }
    }
    else
    {
        string namesToReceive = receiveString(0);
        vector<string> names = stringToVector(namesToReceive);
        string received_phone_numbers = receiveString(0);
        vector<string> phone_numbers = stringToVector(received_phone_numbers);
        searchName = receiveString(0);

        for (int i = 0; i < names.size(); i++)
        {
            check(names[i], phone_numbers[i], searchName, rank);
        }
    }

    double endTime = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);

    printf("Process %d took %f seconds.\n", rank, endTime - startTime);

    MPI_Finalize();
    return 0;
}