#![warn(clippy::default_numeric_fallback)]
#![allow(unused)]
#![allow(clippy::never_loop)]
#![allow(clippy::no_effect)]
#![allow(clippy::unnecessary_operation)]

fn ret_i31() -> i32 {
    23
}

fn concrete_arg(x: i32) {}

fn generic_arg<T>(t: T) {}

struct ConcreteStruct {
    x: i32,
}

struct GenericStruct<T> {
    x: T,
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
    let x = 22;
    let x = 0.12;
    let x: _ = 13;
    let x: [_; 3] = [1, 2, 3];
    let x: (_, i32) = (1, 2);

    let x = if true { (1, 2) } else { (3, 4) };

    let x = match 1 {
        1 => 1,
        _ => 2,
    };

    let x = loop {
        break 1;
    };

    let x = 'outer0: loop {
        {
            'inner0: loop {
                break 3;
            }
        };
        break 2;
    };

    let x = GenericStruct { x: 1 };

    generic_arg(10);
    s.generic_arg(10);
    let f = || -> _ { 1 };

    // Good.
    let x = 22_i32;
    let x: f64 = 0.12;
    let x = 0.12_f64;
    let x: i32 = 13;
    let x: [i32; 3] = [1, 2, 3];
    let x: (i32, i32) = (1, 2);

    let x: (i32, i32) = if true { (1, 2) } else { (3, 4) };

    let x: i32 = match true {
        true => 1,
        _ => 2,
    };

    let x: i32 = loop {
        break 1;
    };

    let x: i32 = 'outer1: loop {
        'inner1: loop {
            break 'outer1 3;
        }
    };

    let x = ConcreteStruct { x: 1 };

    concrete_arg(10);
    s.concrete_arg(10);
    let f = || -> i32 { 1 };
}
