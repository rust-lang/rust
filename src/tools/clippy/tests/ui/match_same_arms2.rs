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

    let x: Result<i32, &str> = Ok(3);

    // No warning because of the guard.
    match x {
        Ok(x) if x * x == 64 => println!("ok"),
        Ok(_) => println!("ok"),
        Err(_) => println!("err"),
    }

    // This used to be a false positive; see issue #1996.
    match x {
        Ok(3) => println!("ok"),
        Ok(x) if x * x == 64 => println!("ok 64"),
        Ok(_) => println!("ok"),
        Err(_) => println!("err"),
    }

    match (x, Some(1i32)) {
        (Ok(x), Some(_)) => println!("ok {}", x),
        (Ok(_), Some(x)) => println!("ok {}", x),
        _ => println!("err"),
    }

    // No warning; different types for `x`.
    match (x, Some(1.0f64)) {
        (Ok(x), Some(_)) => println!("ok {}", x),
        (Ok(_), Some(x)) => println!("ok {}", x),
        _ => println!("err"),
    }

    // False negative #2251.
    match x {
        Ok(_tmp) => println!("ok"),
        Ok(3) => println!("ok"),
        Ok(_) => println!("ok"),
        Err(_) => {
            unreachable!();
        },
    }

    match_expr_like_matches_macro_priority();
}

fn match_expr_like_matches_macro_priority() {
    enum E {
        A,
        B,
        C,
    }
    let x = E::A;
    let _ans = match x {
        E::A => false,
        E::B => false,
        _ => true,
    };
}

fn main() {}
