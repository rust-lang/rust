#![allow(
    clippy::blacklisted_name,
    clippy::collapsible_if,
    clippy::cognitive_complexity,
    clippy::eq_op,
    clippy::needless_continue,
    clippy::needless_return,
    clippy::no_effect,
    clippy::zero_divided_by_zero,
    clippy::unused_unit
)]

fn bar<T>(_: T) {}
fn foo() -> bool {
    unimplemented!()
}

pub enum Abc {
    A,
    B,
    C,
}

#[warn(clippy::match_same_arms)]
#[allow(clippy::unused_unit)]
fn match_same_arms() {
    let _ = match 42 {
        42 => {
            foo();
            let mut a = 42 + [23].len() as i32;
            if true {
                a += 7;
            }
            a = -31 - a;
            a
        },
        _ => {
            //~ ERROR match arms have same body
            foo();
            let mut a = 42 + [23].len() as i32;
            if true {
                a += 7;
            }
            a = -31 - a;
            a
        },
    };

    let _ = match Abc::A {
        Abc::A => 0,
        Abc::B => 1,
        _ => 0, //~ ERROR match arms have same body
    };

    let _ = match 42 {
        42 => foo(),
        51 => foo(), //~ ERROR match arms have same body
        _ => true,
    };

    let _ = match Some(42) {
        Some(_) => 24,
        None => 24, //~ ERROR match arms have same body
    };

    let _ = match Some(42) {
        Some(foo) => 24,
        None => 24,
    };

    let _ = match Some(42) {
        Some(42) => 24,
        Some(a) => 24, // bindings are different
        None => 0,
    };

    let _ = match Some(42) {
        Some(a) if a > 0 => 24,
        Some(a) => 24, // one arm has a guard
        None => 0,
    };

    match (Some(42), Some(42)) {
        (Some(a), None) => bar(a),
        (None, Some(a)) => bar(a), //~ ERROR match arms have same body
        _ => (),
    }

    match (Some(42), Some(42)) {
        (Some(a), ..) => bar(a),
        (.., Some(a)) => bar(a), //~ ERROR match arms have same body
        _ => (),
    }

    match (1, 2, 3) {
        (1, .., 3) => 42,
        (.., 3) => 42, //~ ERROR match arms have same body
        _ => 0,
    };

    let _ = match Some(()) {
        Some(()) => 0.0,
        None => -0.0,
    };

    match (Some(42), Some("")) {
        (Some(a), None) => bar(a),
        (None, Some(a)) => bar(a), // bindings have different types
        _ => (),
    }

    let _ = match 42 {
        42 => 1,
        51 => 1, //~ ERROR match arms have same body
        41 => 2,
        52 => 2, //~ ERROR match arms have same body
        _ => 0,
    };

    let _ = match 42 {
        1 => 2,
        2 => 2, //~ ERROR 2nd matched arms have same body
        3 => 2, //~ ERROR 3rd matched arms have same body
        4 => 3,
        _ => 0,
    };
}

fn main() {}
