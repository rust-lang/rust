// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
