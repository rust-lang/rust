#![feature(custom_attribute)]
#![allow(dead_code, unused_attributes)]

#[miri_run(expected = "Int(1)")]
fn ret() -> i32 {
    1
}

#[miri_run(expected = "Int(-1)")]
fn neg() -> i32 {
    -1
}

#[miri_run(expected = "Int(3)")]
fn add() -> i32 {
    1 + 2
}

#[miri_run(expected = "Int(3)")]
fn indirect_add() -> i32 {
    let x = 1;
    let y = 2;
    x + y
}

#[miri_run(expected = "Int(25)")]
fn arith() -> i32 {
    3*3 + 4*4
}

#[miri_run(expected = "Int(0)")]
fn if_false() -> i32 {
    if false { 1 } else { 0 }
}

#[miri_run(expected = "Int(1)")]
fn if_true() -> i32 {
    if true { 1 } else { 0 }
}

#[miri_run(expected = "Int(2)")]
fn call() -> i32 {
    fn increment(x: i32) -> i32 {
        x + 1
    }

    increment(1)
}

#[miri_run(expected = "Int(3628800)")]
fn factorial_loop() -> i32 {
    let mut product = 1;
    let mut i = 1;

    while i <= 10 {
        product *= i;
        i += 1;
    }

    product
}

#[miri_run(expected = "Int(3628800)")]
fn factorial_recursive() -> i32 {
    fn fact(n: i32) -> i32 {
        if n == 0 {
            1
        } else {
            n * fact(n - 1)
        }
    }

    fact(10)
}

#[miri_run(expected = "Int(1)")]
fn match_bool() -> i32 {
    let b = true;
    match b {
        true => 1,
        false => 0,
    }
}

#[miri_run(expected = "Int(20)")]
fn match_int() -> i32 {
    let n = 2;
    match n {
        0 => 0,
        1 => 10,
        2 => 20,
        3 => 30,
        _ => 100,
    }
}

// #[miri_run(expected = "Int(4)")]
// fn match_int_range() -> i32 {
//     let n = 42;
//     match n {
//         0...9 => 0,
//         10...19 => 1,
//         20...29 => 2,
//         30...39 => 3,
//         40...49 => 4,
//         _ => 5,
//     }
// }

enum MyOption<T> {
    Some { data: T },
    None,
}

#[miri_run(expected = "Int(13)")]
fn match_opt_some() -> i32 {
    let x = MyOption::Some { data: 13 };
    match x {
        MyOption::Some { data } => data,
        MyOption::None => 42,
    }
}

// #[miri_run(expected = "Int(42)")]
// fn match_opt_none() -> i32 {
//     let x = MyOption::None;
//     match x {
//         MyOption::Some { data } => data,
//         MyOption::None => 42,
//     }
// }

/// Test calling a very simple function from the standard library.
#[miri_run(expected = "Int(1)")]
fn cross_crate_fn_call() -> i32 {
    if 1i32.is_positive() { 1 } else { 0 }
}

fn main() {}
