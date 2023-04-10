#include <stdio.h>
#include <stdint.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;
typedef uint32_t u32;

EXPORT i32 add_one_int(i32 x) {
  return 2 + x;
}

EXPORT void printer() {
  printf("printing from C\n");
}

// function with many arguments, to test functionality when some args are stored
// on the stack
EXPORT i32 test_stack_spill(i32 a, i32 b, i32 c, i32 d, i32 e, i32 f, i32 g, i32 h, i32 i, i32 j, i32 k, i32 l) {
  return a+b+c+d+e+f+g+h+i+j+k+l;
}

EXPORT u32 get_unsigned_int() {
  return -10;
}

EXPORT i16 add_int16(i16 x) {
  return x + 3;
}

EXPORT i64 add_short_to_long(i16 x, i64 y) {
  return x + y;
}

EXPORT i32 single_deref(const i32 *p) {
    return *p;
}

EXPORT i32 double_deref(const i32 **p) {
    fprintf(stderr, "%p\n", *p);
    return **p;
}

struct Foo {
    i32 a;
    float b;
};

EXPORT i32 struct_int(const struct Foo* p) {
    return p->a;
}

EXPORT float struct_float(const struct Foo* p) {
    return p->b;
}
