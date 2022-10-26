#include <stdio.h>

__declspec(dllexport) void extern_fn_1() {
    printf("extern_fn_1\n");
    fflush(stdout);
}

__declspec(dllexport) void extern_fn_2() {
    printf("extern_fn_2 in extern_1\n");
    fflush(stdout);
}
