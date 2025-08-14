#include <stdio.h>
#include <stdint.h>

void _cdecl LooksLikeAPrivateGlobal(int i) {
    printf("LooksLikeAPrivateGlobal(%d)\n", i);
    fflush(stdout);
}

void _cdecl cdecl_fn_undecorated(int i) {
    printf("cdecl_fn_undecorated(%d)\n", i);
    fflush(stdout);
}

void _cdecl cdecl_fn_undecorated2(int i) {
    printf("cdecl_fn_undecorated2(%d)\n", i);
    fflush(stdout);
}

void _cdecl cdecl_fn_noprefix(int i) {
    printf("cdecl_fn_noprefix(%d)\n", i);
    fflush(stdout);
}

void _cdecl cdecl_fn_decorated(int i) {
    printf("cdecl_fn_decorated(%d)\n", i);
    fflush(stdout);
}

void __stdcall stdcall_fn_undecorated(int i) {
    printf("stdcall_fn_undecorated(%d)\n", i);
    fflush(stdout);
}

void __stdcall stdcall_fn_undecorated2(int i) {
    printf("stdcall_fn_undecorated2(%d)\n", i);
    fflush(stdout);
}

void __stdcall stdcall_fn_noprefix(int i) {
    printf("stdcall_fn_noprefix(%d)\n", i);
    fflush(stdout);
}

void __stdcall stdcall_fn_decorated(int i) {
    printf("stdcall_fn_decorated(%d)\n", i);
    fflush(stdout);
}

void __fastcall fastcall_fn_undecorated(int i) {
    printf("fastcall_fn_undecorated(%d)\n", i);
    fflush(stdout);
}

void __fastcall fastcall_fn_undecorated2(int i) {
    printf("fastcall_fn_undecorated2(%d)\n", i);
    fflush(stdout);
}

void __fastcall fastcall_fn_noprefix(int i) {
    printf("fastcall_fn_noprefix(%d)\n", i);
    fflush(stdout);
}

void __fastcall fastcall_fn_decorated(int i) {
    printf("fastcall_fn_decorated(%d)\n", i);
    fflush(stdout);
}

int extern_variable_undecorated = 0;
__declspec(dllexport) void print_extern_variable_undecorated() {
    printf("extern_variable_undecorated value: %d\n", extern_variable_undecorated);
    fflush(stdout);
}

int extern_variable_noprefix = 0;
__declspec(dllexport) void print_extern_variable_noprefix() {
    printf("extern_variable_noprefix value: %d\n", extern_variable_noprefix);
    fflush(stdout);
}

int extern_variable_decorated = 0;
__declspec(dllexport) void print_extern_variable_decorated() {
    printf("extern_variable_decorated value: %d\n", extern_variable_decorated);
    fflush(stdout);
}

// GCC doesn't support vectorcall: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=89485
#ifdef _MSC_VER
void __vectorcall vectorcall_fn_undecorated(int i) {
    printf("vectorcall_fn_undecorated(%d)\n", i);
    fflush(stdout);
}

void __vectorcall vectorcall_fn_undecorated2(int i) {
    printf("vectorcall_fn_undecorated2(%d)\n", i);
    fflush(stdout);
}

void __vectorcall vectorcall_fn_noprefix(int i) {
    printf("vectorcall_fn_noprefix(%d)\n", i);
    fflush(stdout);
}

void __vectorcall vectorcall_fn_decorated(int i) {
    printf("vectorcall_fn_decorated(%d)\n", i);
    fflush(stdout);
}
#endif
