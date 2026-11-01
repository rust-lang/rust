// Warn when we encounter a manual `Default` impl that could be derived.
// Restricted only to types using `default_field_values`.
#![feature(default_field_values)]
#![allow(dead_code)]
#![deny(default_overrides_default_fields)]
struct S(i32);
fn s() -> S { S(1) }

struct A {
    x: S,
    y: i32 = 1,
}

impl Default for A { //~ ERROR default_overrides_default_fields
    fn default() -> Self {
        A {
            y: 0,
            x: s(),
        }
    }
}

struct B {
    x: S = S(3),
    y: i32 = 1,
}

impl Default for B { //~ ERROR default_overrides_default_fields
    fn default() -> Self {
        B {
            x: s(),
            y: 0,
        }
    }
}

struct C {
    x: S,
    y: i32 = 1,
    z: i32 = 1,
}

impl Default for C { //~ ERROR default_overrides_default_fields
    fn default() -> Self {
        C {
            x: s(),
            y: 0,
            ..
        }
    }
}

struct D {
    x: S,
    y: i32 = 1,
    z: i32 = 1,
}

impl Default for D { //~ ERROR default_overrides_default_fields
    fn default() -> Self {
        D {
            y: 0,
            x: s(),
            ..
        }
    }
}

struct E {
    x: S,
    y: i32 = 1,
    z: i32 = 1,
}

impl Default for E { //~ ERROR default_overrides_default_fields
    fn default() -> Self {
        E {
            y: 0,
            z: 0,
            x: s(),
        }
    }
}

// Let's ensure that the span for `x` and the span for `y` don't overlap when suggesting their
// removal in favor of their default field values.
struct E2 {
    x: S,
    y: i32 = 1,
    z: i32 = 1,
}

impl Default for E2 { //~ ERROR default_overrides_default_fields
    fn default() -> Self {
        E2 {
            x: s(),
            y: i(),
            z: 0,
        }
    }
}

fn i() -> i32 {
    1
}

// Account for a `const fn` being the `Default::default()` of a field's type.
struct F {
    x: G,
    y: i32 = 1,
}

impl Default for F { //~ ERROR default_overrides_default_fields
    fn default() -> Self {
        F {
            x: g_const(),
            y: 0,
        }
    }
}

struct G;

impl Default for G { // ok
    fn default() -> Self {
        g_const()
    }
}

const fn g_const() -> G {
    G
}

// Account for a `const fn` being used in `Default::default()`, even if the type doesn't use it as
// its own `Default`. We suggest setting the default field value in that case.
struct H {
    x: I,
    y: i32 = 1,
}

impl Default for H { //~ ERROR default_overrides_default_fields
    fn default() -> Self {
        H {
            x: i_const(),
            y: 0,
        }
    }
}

struct I;

const fn i_const() -> I {
    I
}

// Account for a `const` and struct literal being the `Default::default()` of a field's type.
struct M {
    x: N,
    y: i32 = 1,
    z: A,
}

impl Default for M { // ok, `y` is not specified
    fn default() -> Self {
        M {
            x: N_CONST,
            z: A {
                x: S(0),
                y: 0,
            },
            ..
        }
    }
}

struct N;

const N_CONST: N = N;

struct O {
    x: Option<i32>,
    y: i32 = 1,
}

impl Default for O { //~ ERROR default_overrides_default_fields
    fn default() -> Self {
        O {
            x: None,
            y: 1,
        }
    }
}

fn main() {}
