#include <stdio.h>
#include <stdint.h>

struct S {
    uint8_t x;
    int32_t y;
};

struct S2 {
    int32_t x;
    uint8_t y;
};

struct S3 {
    uint8_t x[5];
};

__declspec(dllexport) void __stdcall stdcall_fn_1(int i) {
    printf("stdcall_fn_1(%d)\n", i);
    fflush(stdout);
}

__declspec(dllexport) void __stdcall stdcall_fn_2(uint8_t i, float f) {
    printf("stdcall_fn_2(%d, %.1f)\n", i, f);
    fflush(stdout);
}

__declspec(dllexport) void __stdcall stdcall_fn_3(double d) {
    printf("stdcall_fn_3(%.1f)\n", d);
    fflush(stdout);
}

__declspec(dllexport) void __stdcall stdcall_fn_4(uint8_t i, uint8_t j, float f) {
    printf("stdcall_fn_4(%d, %d, %.1f)\n", i, j, f);
    fflush(stdout);
}

__declspec(dllexport) void __stdcall stdcall_fn_5(struct S s, int i) {
    printf("stdcall_fn_5(S { x: %d, y: %d }, %d)\n", s.x, s.y, i);
    fflush(stdout);
}

// Test that stdcall support works correctly with the nullable pointer optimization.
__declspec(dllexport) void __stdcall stdcall_fn_6(struct S* s) {
    if (s) {
        printf("stdcall_fn_6(S { x: %d, y: %d })\n", s->x, s->y);
    } else {
        printf("stdcall_fn_6(null)\n");
    }
    fflush(stdout);
}

__declspec(dllexport) void __stdcall stdcall_fn_7(struct S2 s, int i) {
    printf("stdcall_fn_7(S2 { x: %d, y: %d }, %d)\n", s.x, s.y, i);
    fflush(stdout);
}

// Verify that we compute the correct amount of space in the argument list for a 5-byte struct.
__declspec(dllexport) void __stdcall stdcall_fn_8(struct S3 s, struct S3 t) {
    printf("stdcall_fn_8(S3 { x: [%d, %d, %d, %d, %d] }, S3 { x: [%d, %d, %d, %d, %d] })\n",
           s.x[0], s.x[1], s.x[2], s.x[3], s.x[4],
           t.x[0], t.x[1], t.x[2], t.x[3], t.x[4]
        );
    fflush(stdout);
}

// test whether f64/double values are aligned on 4-byte or 8-byte boundaries.
__declspec(dllexport) void __stdcall stdcall_fn_9(uint8_t x, double y) {
    printf("stdcall_fn_9(%d, %.1f)\n", x, y);
    fflush(stdout);
}

__declspec(dllexport) void __stdcall stdcall_fn_10(int i) {
    printf("stdcall_fn_10(%d)\n", i);
    fflush(stdout);
}

__declspec(dllexport) void __fastcall fastcall_fn_1(int i) {
    printf("fastcall_fn_1(%d)\n", i);
    fflush(stdout);
}

__declspec(dllexport) void __fastcall fastcall_fn_2(uint8_t i, float f) {
    printf("fastcall_fn_2(%d, %.1f)\n", i, f);
    fflush(stdout);
}

__declspec(dllexport) void __fastcall fastcall_fn_3(double d) {
    printf("fastcall_fn_3(%.1f)\n", d);
    fflush(stdout);
}

__declspec(dllexport) void __fastcall fastcall_fn_4(uint8_t i, uint8_t j, float f) {
    printf("fastcall_fn_4(%d, %d, %.1f)\n", i, j, f);
    fflush(stdout);
}

__declspec(dllexport) void __fastcall fastcall_fn_5(struct S s, int i) {
    printf("fastcall_fn_5(S { x: %d, y: %d }, %d)\n", s.x, s.y, i);
    fflush(stdout);
}

__declspec(dllexport) void __fastcall fastcall_fn_6(struct S* s) {
    if (s) {
        printf("fastcall_fn_6(S { x: %d, y: %d })\n", s->x, s->y);
    } else {
        printf("fastcall_fn_6(null)\n");
    }
    fflush(stdout);
}

__declspec(dllexport) void __fastcall fastcall_fn_7(struct S2 s, int i) {
    printf("fastcall_fn_7(S2 { x: %d, y: %d }, %d)\n", s.x, s.y, i);
    fflush(stdout);
}

__declspec(dllexport) void __fastcall fastcall_fn_8(struct S3 s, struct S3 t) {
    printf("fastcall_fn_8(S3 { x: [%d, %d, %d, %d, %d] }, S3 { x: [%d, %d, %d, %d, %d] })\n",
           s.x[0], s.x[1], s.x[2], s.x[3], s.x[4],
           t.x[0], t.x[1], t.x[2], t.x[3], t.x[4]
        );
    fflush(stdout);
}

__declspec(dllexport) void __fastcall fastcall_fn_9(uint8_t x, double y) {
    printf("fastcall_fn_9(%d, %.1f)\n", x, y);
    fflush(stdout);
}

__declspec(dllexport) void __fastcall fastcall_fn_10(int i) {
    printf("fastcall_fn_10(%d)\n", i);
    fflush(stdout);
}

// GCC doesn't support vectorcall: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=89485
#ifdef _MSC_VER
__declspec(dllexport) void __vectorcall vectorcall_fn_1(int i) {
    printf("vectorcall_fn_1(%d)\n", i);
    fflush(stdout);
}

__declspec(dllexport) void __vectorcall vectorcall_fn_2(uint8_t i, float f) {
    printf("vectorcall_fn_2(%d, %.1f)\n", i, f);
    fflush(stdout);
}

__declspec(dllexport) void __vectorcall vectorcall_fn_3(double d) {
    printf("vectorcall_fn_3(%.1f)\n", d);
    fflush(stdout);
}

__declspec(dllexport) void __vectorcall vectorcall_fn_4(uint8_t i, uint8_t j, float f) {
    printf("vectorcall_fn_4(%d, %d, %.1f)\n", i, j, f);
    fflush(stdout);
}

__declspec(dllexport) void __vectorcall vectorcall_fn_5(struct S s, int i) {
    printf("vectorcall_fn_5(S { x: %d, y: %d }, %d)\n", s.x, s.y, i);
    fflush(stdout);
}

__declspec(dllexport) void __vectorcall vectorcall_fn_6(struct S* s) {
    if (s) {
        printf("vectorcall_fn_6(S { x: %d, y: %d })\n", s->x, s->y);
    } else {
        printf("vectorcall_fn_6(null)\n");
    }
    fflush(stdout);
}

__declspec(dllexport) void __vectorcall vectorcall_fn_7(struct S2 s, int i) {
    printf("vectorcall_fn_7(S2 { x: %d, y: %d }, %d)\n", s.x, s.y, i);
    fflush(stdout);
}

__declspec(dllexport) void __vectorcall vectorcall_fn_8(struct S3 s, struct S3 t) {
    printf("vectorcall_fn_8(S3 { x: [%d, %d, %d, %d, %d] }, S3 { x: [%d, %d, %d, %d, %d] })\n",
           s.x[0], s.x[1], s.x[2], s.x[3], s.x[4],
           t.x[0], t.x[1], t.x[2], t.x[3], t.x[4]
        );
    fflush(stdout);
}

__declspec(dllexport) void __vectorcall vectorcall_fn_9(uint8_t x, double y) {
    printf("vectorcall_fn_9(%d, %.1f)\n", x, y);
    fflush(stdout);
}

__declspec(dllexport) void __vectorcall vectorcall_fn_10(int i) {
    printf("vectorcall_fn_10(%d)\n", i);
    fflush(stdout);
}
#endif
