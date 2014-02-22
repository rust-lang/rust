// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(macro_rules)];

macro_rules! assert_approx_eq(
    ($a:expr, $b:expr) => ({
        let (a, b) = (&$a, &$b);
        fail_unless!((*a - *b).abs() < 1.0e-6,
                "{} is not approximately equal to {}", *a, *b);
    })
)

static A: int = -4 + 3;
static A2: uint = 3 + 3;
static B: f64 = 3.0 + 2.7;

static C: int = 3 - 4;
static D: uint = 3 - 3;
static E: f64 = 3.0 - 2.7;

static E2: int = -3 * 3;
static F: uint = 3 * 3;
static G: f64 = 3.3 * 3.3;

static H: int = 3 / -1;
static I: uint = 3 / 3;
static J: f64 = 3.3 / 3.3;

static N: bool = true && false;

static O: bool = true || false;

static P: int = 3 & 1;
static Q: uint = 1 & 3;

static R: int = 3 | 1;
static S: uint = 1 | 3;

static T: int = 3 ^ 1;
static U: uint = 1 ^ 3;

static V: int = 1 << 3;

// NOTE: better shr coverage
static W: int = 1024 >> 4;
static X: uint = 1024 >> 4;

static Y: bool = 1 == 1;
static Z: bool = 1.0 == 1.0;

static AA: bool = 1 <= 2;
static AB: bool = -1 <= 2;
static AC: bool = 1.0 <= 2.0;

static AD: bool = 1 < 2;
static AE: bool = -1 < 2;
static AF: bool = 1.0 < 2.0;

static AG: bool = 1 != 2;
static AH: bool = -1 != 2;
static AI: bool = 1.0 != 2.0;

static AJ: bool = 2 >= 1;
static AK: bool = 2 >= -2;
static AL: bool = 1.0 >= -2.0;

static AM: bool = 2 > 1;
static AN: bool = 2 > -2;
static AO: bool = 1.0 > -2.0;

pub fn main() {
    fail_unless_eq!(A, -1);
    fail_unless_eq!(A2, 6);
    assert_approx_eq!(B, 5.7);

    fail_unless_eq!(C, -1);
    fail_unless_eq!(D, 0);
    assert_approx_eq!(E, 0.3);

    fail_unless_eq!(E2, -9);
    fail_unless_eq!(F, 9);
    assert_approx_eq!(G, 10.89);

    fail_unless_eq!(H, -3);
    fail_unless_eq!(I, 1);
    assert_approx_eq!(J, 1.0);

    fail_unless_eq!(N, false);

    fail_unless_eq!(O, true);

    fail_unless_eq!(P, 1);
    fail_unless_eq!(Q, 1);

    fail_unless_eq!(R, 3);
    fail_unless_eq!(S, 3);

    fail_unless_eq!(T, 2);
    fail_unless_eq!(U, 2);

    fail_unless_eq!(V, 8);

    fail_unless_eq!(W, 64);
    fail_unless_eq!(X, 64);

    fail_unless_eq!(Y, true);
    fail_unless_eq!(Z, true);

    fail_unless_eq!(AA, true);
    fail_unless_eq!(AB, true);
    fail_unless_eq!(AC, true);

    fail_unless_eq!(AD, true);
    fail_unless_eq!(AE, true);
    fail_unless_eq!(AF, true);

    fail_unless_eq!(AG, true);
    fail_unless_eq!(AH, true);
    fail_unless_eq!(AI, true);

    fail_unless_eq!(AJ, true);
    fail_unless_eq!(AK, true);
    fail_unless_eq!(AL, true);

    fail_unless_eq!(AM, true);
    fail_unless_eq!(AN, true);
    fail_unless_eq!(AO, true);
}
