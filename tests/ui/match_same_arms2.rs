#![warn(clippy::match_same_arms)]
#![allow(clippy::blacklisted_name)]

fn bar<T>(_: T) {}
fn foo() -> bool {
    unimplemented!()
}

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

    let _ = match Some(()) {
        Some(()) => 0.0,
        None => -0.0,
    };

    match (Some(42), Some("")) {
        (Some(a), None) => bar(a),
        (None, Some(a)) => bar(a), // bindings have different types
        _ => (),
    }
}

fn main() {}
