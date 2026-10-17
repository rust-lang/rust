#include <stdio.h>

void __stdcall exported_function_stdcall(int i) {
    printf("exported_function_stdcall(%d)\n", i);
    fflush(stdout);
}

void __fastcall exported_function_fastcall(int i) {
    printf("exported_function_fastcall(%d)\n", i);
    fflush(stdout);
}
