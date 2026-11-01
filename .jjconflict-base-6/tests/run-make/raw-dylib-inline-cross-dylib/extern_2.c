#include <stdio.h>

__declspec(dllexport) void extern_fn_2() {
    printf("extern_fn_2 in extern_2\n");
    fflush(stdout);
}
