# Source Code Representation

This part describes the process of taking raw source code from the user and
transforming it into various forms that the compiler can work with easily.
These are called _intermediate representations (IRs)_.

This process starts with compiler understanding what the user has asked for:
parsing the command line arguments given and determining what it is to compile.
After that, the compiler transforms the user input into a series of IRs that
look progressively less like what the user wrote.
