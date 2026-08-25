#include <stdio.h>

__declspec(dllexport) int extern_variable = 0;

__declspec(dllexport) void extern_fn_1() {
    printf("extern_fn_1\n");
    fflush(stdout);
}

__declspec(dllexport) void extern_fn_2() {
    printf("extern_fn_2; didn't get the rename\n");
    fflush(stdout);
}

__declspec(dllexport) void print_extern_variable() {
    printf("extern_variable value: %d\n", extern_variable);
    fflush(stdout);
}

__declspec(dllexport) void extern_fn_with_long_name() {
    printf("extern_fn_with_long_name; got the rename\n");
    fflush(stdout);
}

__declspec(dllexport) void extern_fn_4() {
    printf("extern_fn_4\n");
    fflush(stdout);
}
