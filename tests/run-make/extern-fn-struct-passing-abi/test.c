#include <assert.h>
#include <stdint.h>

struct Rect {
    int32_t a;
    int32_t b;
    int32_t c;
    int32_t d;
};

struct BiggerRect {
    struct Rect s;
    int32_t a;
    int32_t b;
};

struct FloatRect {
    int32_t a;
    int32_t b;
    double c;
};

struct Huge {
    int32_t a;
    int32_t b;
    int32_t c;
    int32_t d;
    int32_t e;
};

struct Huge64 {
    int64_t a;
    int64_t b;
    int64_t c;
    int64_t d;
    int64_t e;
};

struct FloatPoint {
    double x;
    double y;
};

struct FloatOne {
    double x;
};

struct IntOdd {
    int8_t a;
    int8_t b;
    int8_t c;
};

// System V x86_64 ABI:
// a, b, c, d, e should be in registers
// s should be byval pointer
//
// Win64 ABI:
// a, b, c, d should be in registers
// e should be on the stack
// s should be byval pointer
void byval_rect(int32_t a, int32_t b, int32_t c, int32_t d, int32_t e, struct Rect s) {
    assert(a == 1);
    assert(b == 2);
    assert(c == 3);
    assert(d == 4);
    assert(e == 5);
    assert(s.a == 553);
    assert(s.b == 554);
    assert(s.c == 555);
    assert(s.d == 556);
}

// System V x86_64 ABI:
// a, b, c, d, e, f should be in registers
// s should be byval pointer on the stack
//
// Win64 ABI:
// a, b, c, d should be in registers
// e, f should be on the stack
// s should be byval pointer on the stack
void byval_many_rect(int32_t a, int32_t b, int32_t c, int32_t d, int32_t e,
                     int32_t f, struct Rect s) {
    assert(a == 1);
    assert(b == 2);
    assert(c == 3);
    assert(d == 4);
    assert(e == 5);
    assert(f == 6);
    assert(s.a == 553);
    assert(s.b == 554);
    assert(s.c == 555);
    assert(s.d == 556);
}

// System V x86_64 ABI:
// a, b, c, d, e, f, g should be in sse registers
// s should be split across 2 registers
// t should be byval pointer
//
// Win64 ABI:
// a, b, c, d should be in sse registers
// e, f, g should be on the stack
// s should be on the stack (treated as 2 i64's)
// t should be on the stack (treated as an i64 and a double)
void byval_rect_floats(float a, float b, double c, float d, float e,
                       float f, double g, struct Rect s, struct FloatRect t) {
    assert(a == 1.);
    assert(b == 2.);
    assert(c == 3.);
    assert(d == 4.);
    assert(e == 5.);
    assert(f == 6.);
    assert(g == 7.);
    assert(s.a == 553);
    assert(s.b == 554);
    assert(s.c == 555);
    assert(s.d == 556);
    assert(t.a == 3489);
    assert(t.b == 3490);
    assert(t.c == 8.);
}

// System V x86_64 ABI:
// a, b, d, e, f should be in registers
// c passed via sse registers
// s should be byval pointer
//
// Win64 ABI:
// a, b, d should be in registers
// c passed via sse registers
// e, f should be on the stack
// s should be byval pointer
void byval_rect_with_float(int32_t a, int32_t b, float c, int32_t d,
                           int32_t e, int32_t f, struct Rect s) {
    assert(a == 1);
    assert(b == 2);
    assert(c == 3.);
    assert(d == 4);
    assert(e == 5);
    assert(f == 6);
    assert(s.a == 553);
    assert(s.b == 554);
    assert(s.c == 555);
    assert(s.d == 556);
}

// System V x86_64 ABI:
// a, b, d, e, f should be byval pointer (on the stack)
// g passed via register (fixes #41375)
//
// Win64 ABI:
// a, b, d, e, f, g should be byval pointer
void byval_rect_with_many_huge(struct Huge a, struct Huge b, struct Huge c,
                               struct Huge d, struct Huge e, struct Huge f,
                               struct Rect g) {
    assert(g.a == 123);
    assert(g.b == 456);
    assert(g.c == 789);
    assert(g.d == 420);
}

// System V x86_64 ABI:
// a, b, d, e, f should be byval pointer (on the stack)
// g passed via register (fixes #41375)
//
// i686-windows ABI:
// a, b, d, e, f, g should be byval pointer
void byval_rect_with_many_huge64(struct Huge64 a, struct Huge64 b, struct Huge64 c,
                               struct Huge64 d, struct Huge64 e, struct Huge64 f,
                               struct Rect g) {
    assert(g.a == 1234);
    assert(g.b == 4567);
    assert(g.c == 7890);
    assert(g.d == 4209);
}

// System V x86_64 & Win64 ABI:
// a, b should be in registers
// s should be split across 2 integer registers
void split_rect(int32_t a, int32_t b, struct Rect s) {
    assert(a == 1);
    assert(b == 2);
    assert(s.a == 553);
    assert(s.b == 554);
    assert(s.c == 555);
    assert(s.d == 556);
}

// System V x86_64 & Win64 ABI:
// a, b should be in sse registers
// s should be split across integer & sse registers
void split_rect_floats(float a, float b, struct FloatRect s) {
    assert(a == 1.);
    assert(b == 2.);
    assert(s.a == 3489);
    assert(s.b == 3490);
    assert(s.c == 8.);
}

// System V x86_64 ABI:
// a, b, d, f should be in registers
// c, e passed via sse registers
// s should be split across 2 registers
//
// Win64 ABI:
// a, b, d should be in registers
// c passed via sse registers
// e, f should be on the stack
// s should be on the stack (treated as 2 i64's)
void split_rect_with_floats(int32_t a, int32_t b, float c,
                            int32_t d, float e, int32_t f, struct Rect s) {
    assert(a == 1);
    assert(b == 2);
    assert(c == 3.);
    assert(d == 4);
    assert(e == 5.);
    assert(f == 6);
    assert(s.a == 553);
    assert(s.b == 554);
    assert(s.c == 555);
    assert(s.d == 556);
}

// System V x86_64 & Win64 ABI:
// a, b, c should be in registers
// s should be split across 2 registers
// t should be a byval pointer
void split_and_byval_rect(int32_t a, int32_t b, int32_t c, struct Rect s, struct Rect t) {
    assert(a == 1);
    assert(b == 2);
    assert(c == 3);
    assert(s.a == 553);
    assert(s.b == 554);
    assert(s.c == 555);
    assert(s.d == 556);
    assert(t.a == 553);
    assert(t.b == 554);
    assert(t.c == 555);
    assert(t.d == 556);
}

// System V x86_64 & Win64 ABI:
// a, b should in registers
// s and return should be split across 2 registers
struct Rect split_ret_byval_struct(int32_t a, int32_t b, struct Rect s) {
    assert(a == 1);
    assert(b == 2);
    assert(s.a == 553);
    assert(s.b == 554);
    assert(s.c == 555);
    assert(s.d == 556);
    return s;
}

// System V x86_64 & Win64 ABI:
// a, b, c, d should be in registers
// return should be in a hidden sret pointer
// s should be a byval pointer
struct BiggerRect sret_byval_struct(int32_t a, int32_t b, int32_t c, int32_t d, struct Rect s) {
    assert(a == 1);
    assert(b == 2);
    assert(c == 3);
    assert(d == 4);
    assert(s.a == 553);
    assert(s.b == 554);
    assert(s.c == 555);
    assert(s.d == 556);

    struct BiggerRect t;
    t.s = s; t.a = 27834; t.b = 7657;
    return t;
}

// System V x86_64 & Win64 ABI:
// a, b should be in registers
// return should be in a hidden sret pointer
// s should be split across 2 registers
struct BiggerRect sret_split_struct(int32_t a, int32_t b, struct Rect s) {
    assert(a == 1);
    assert(b == 2);
    assert(s.a == 553);
    assert(s.b == 554);
    assert(s.c == 555);
    assert(s.d == 556);

    struct BiggerRect t;
    t.s = s; t.a = 27834; t.b = 7657;
    return t;
}

// System V x86_64 & Win64 ABI:
// s should be byval pointer (since sizeof(s) > 16)
// return should in a hidden sret pointer
struct Huge huge_struct(struct Huge s) {
    assert(s.a == 5647);
    assert(s.b == 5648);
    assert(s.c == 5649);
    assert(s.d == 5650);
    assert(s.e == 5651);

    return s;
}

// System V x86_64 & i686-windows ABI:
// s should be byval pointer
// return should in a hidden sret pointer
struct Huge64 huge64_struct(struct Huge64 s) {
    assert(s.a == 1234);
    assert(s.b == 1335);
    assert(s.c == 1436);
    assert(s.d == 1537);
    assert(s.e == 1638);

    return s;
}

// System V x86_64 ABI:
// p should be in registers
// return should be in registers
//
// Win64 ABI and 64-bit PowerPC ELFv1 ABI:
// p should be a byval pointer
// return should be in a hidden sret pointer
struct FloatPoint float_point(struct FloatPoint p) {
    assert(p.x == 5.);
    assert(p.y == -3.);

    return p;
}

// 64-bit PowerPC ELFv1 ABI:
// f1 should be in a register
// return should be in a hidden sret pointer
struct FloatOne float_one(struct FloatOne f1) {
    assert(f1.x == 7.);

    return f1;
}

// 64-bit PowerPC ELFv1 ABI:
// i should be in the least-significant bits of a register
// return should be in a hidden sret pointer
struct IntOdd int_odd(struct IntOdd i) {
    assert(i.a == 1);
    assert(i.b == 2);
    assert(i.c == 3);

    return i;
}
