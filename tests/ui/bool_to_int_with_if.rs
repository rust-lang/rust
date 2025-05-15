#![warn(clippy::bool_to_int_with_if)]
#![allow(unused, dead_code, clippy::unnecessary_operation, clippy::no_effect)]

fn main() {
    let a = true;
    let b = false;

    let x = 1;
    let y = 2;

    // Should lint
    // precedence
    if a {
        //~^ bool_to_int_with_if
        1
    } else {
        0
    };
    if a {
        //~^ bool_to_int_with_if
        0
    } else {
        1
    };
    if !a {
        //~^ bool_to_int_with_if
        1
    } else {
        0
    };
    if a || b {
        //~^ bool_to_int_with_if
        1
    } else {
        0
    };
    if cond(a, b) {
        //~^ bool_to_int_with_if
        1
    } else {
        0
    };
    if x + y < 4 {
        //~^ bool_to_int_with_if
        1
    } else {
        0
    };

    // if else if
    if a {
        123
    } else if b {
        //~^ bool_to_int_with_if
        1
    } else {
        0
    };

    // if else if inverted
    if a {
        123
    } else if b {
        //~^ bool_to_int_with_if
        0
    } else {
        1
    };

    // Shouldn't lint

    if a {
        1
    } else if b {
        0
    } else {
        3
    };

    if a {
        3
    } else if b {
        1
    } else {
        -2
    };

    if a {
        3
    } else {
        0
    };
    if a {
        side_effect();
        1
    } else {
        0
    };
    if a {
        1
    } else {
        side_effect();
        0
    };

    // multiple else ifs
    if a {
        123
    } else if b {
        1
    } else if a | b {
        0
    } else {
        123
    };

    pub const SHOULD_NOT_LINT: usize = if true { 1 } else { 0 };

    // https://github.com/rust-lang/rust-clippy/issues/10452
    let should_not_lint = [(); if true { 1 } else { 0 }];

    let should_not_lint = const { if true { 1 } else { 0 } };

    some_fn(a);
}

// Lint returns and type inference
fn some_fn(a: bool) -> u8 {
    if a { 1 } else { 0 }
    //~^ bool_to_int_with_if
}

fn side_effect() {}

fn cond(a: bool, b: bool) -> bool {
    a || b
}

enum Enum {
    A,
    B,
}

fn if_let(a: Enum, b: Enum) {
    if let Enum::A = a {
        1
    } else {
        0
    };

    if let Enum::A = a
        && let Enum::B = b
    {
        1
    } else {
        0
    };
}

fn issue14628() {
    macro_rules! mac {
        (if $cond:expr, $then:expr, $else:expr) => {
            if $cond { $then } else { $else }
        };
        (zero) => {
            0
        };
        (one) => {
            1
        };
    }

    let _ = if dbg!(4 > 0) { 1 } else { 0 };
    //~^ bool_to_int_with_if

    let _ = dbg!(if 4 > 0 { 1 } else { 0 });
    //~^ bool_to_int_with_if

    let _ = mac!(if 4 > 0, 1, 0);
    let _ = if 4 > 0 { mac!(one) } else { 0 };
    let _ = if 4 > 0 { 1 } else { mac!(zero) };
}
