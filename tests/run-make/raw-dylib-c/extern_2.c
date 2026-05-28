#include <stdio.h>

__declspec(dllexport) void extern_fn_3() {
    printf("extern_fn_3\n");
    fflush(stdout);
}
