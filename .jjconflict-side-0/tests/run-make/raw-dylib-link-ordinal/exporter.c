#include <stdio.h>

void exported_function() {
    printf("exported_function\n");
}

int exported_variable = 0;

void print_exported_variable() {
    printf("exported_variable value: %d\n", exported_variable);
    fflush(stdout);
}
