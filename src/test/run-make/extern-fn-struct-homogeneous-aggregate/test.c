// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test calls to c functions with various lengths of homogeneous
// floating point aggregates. Lengths are chosen to hit corner cases on
// the ppc64 and aarch64 parameter passing ABIs.

#include <assert.h>

struct TwoFloats {
    float a, b;
};

struct FourFloats {
    float a, b, c, d;
};

struct EightFloats {
    float a, b, c, d, e, f, g, h;
};

struct NineFloats {
    float a, b, c, d, e, f, g, h, i;
};

struct TwoFloats test_two_floats(struct TwoFloats a)
{
    struct TwoFloats b;

    assert(a.a == 2001);
    assert(a.b == 2002);

    b.a = 2001;
    b.b = 2002;

    return b;
}

struct FourFloats test_four_floats(struct FourFloats a)
{
    struct FourFloats b;

    assert(a.a == 4001);
    assert(a.b == 4002);
    assert(a.c == 4003);
    assert(a.d == 4004);

    b.a = 4001;
    b.b = 4002;
    b.c = 4003;
    b.d = 4004;

    return b;
}

struct EightFloats test_eight_floats(struct EightFloats a)
{
    struct EightFloats b;

    assert(a.a == 8001);
    assert(a.b == 8002);
    assert(a.c == 8003);
    assert(a.d == 8004);
    assert(a.e == 8005);
    assert(a.f == 8006);
    assert(a.g == 8007);
    assert(a.h == 8008);

    b.a = 8001;
    b.b = 8002;
    b.c = 8003;
    b.d = 8004;
    b.e = 8005;
    b.f = 8006;
    b.g = 8007;
    b.h = 8008;

    return b;
}

struct NineFloats test_nine_floats(struct NineFloats a)
{
    struct NineFloats b;

    assert(a.a == 9001);
    assert(a.b == 9002);
    assert(a.c == 9003);
    assert(a.d == 9004);
    assert(a.e == 9005);
    assert(a.f == 9006);
    assert(a.g == 9007);
    assert(a.h == 9008);
    assert(a.i == 9009);

    b.a = 9001;
    b.b = 9002;
    b.c = 9003;
    b.d = 9004;
    b.e = 9005;
    b.f = 9006;
    b.g = 9007;
    b.h = 9008;
    b.i = 9009;

    return b;
}
