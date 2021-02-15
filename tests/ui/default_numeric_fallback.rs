#![warn(clippy::default_numeric_fallback)]
#![allow(unused)]
#![allow(clippy::never_loop)]
#![allow(clippy::no_effect)]
#![allow(clippy::unnecessary_operation)]

fn concrete_arg(x: i32) {}

fn generic_arg<T>(t: T) {}

struct ConcreteStruct {
    x: i32,
}

struct StructForMethodCallTest {
    x: i32,
}

impl StructForMethodCallTest {
    fn concrete_arg(&self, x: i32) {}

    fn generic_arg<T>(&self, t: T) {}
}

fn main() {
    let s = StructForMethodCallTest { x: 10_i32 };

    // Bad.
    let x = 1;
    let x = 0.1;

    let x = if true { 1 } else { 2 };

    let x: _ = {
        let y = 1;
        1
    };

    generic_arg(10);
    s.generic_arg(10);
    let x: _ = generic_arg(10);
    let x: _ = s.generic_arg(10);

    // Good.
    let x = 1_i32;
    let x: i32 = 1;
    let x: _ = 1;
    let x = 0.1_f64;
    let x: f64 = 0.1;
    let x: _ = 0.1;

    let x: _ = if true { 1 } else { 2 };

    concrete_arg(10);
    s.concrete_arg(10);
    let x = concrete_arg(10);
    let x = s.concrete_arg(10);
}
