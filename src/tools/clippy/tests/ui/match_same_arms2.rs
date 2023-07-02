#![warn(clippy::match_same_arms)]
#![allow(
    clippy::disallowed_names,
    clippy::diverging_sub_expression,
    clippy::uninlined_format_args
)]

fn bar<T>(_: T) {}
fn foo() -> bool {
    unimplemented!()
}

fn match_same_arms() {
    let _ = match 42 {
        42 => {
            //~^ ERROR: this match arm has an identical body to the `_` wildcard arm
            foo();
            let mut a = 42 + [23].len() as i32;
            if true {
                a += 7;
            }
            a = -31 - a;
            a
        },
        _ => {
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
        51 => foo(), //~ ERROR: this match arm has an identical body to another arm
        _ => true,
    };

    let _ = match Some(42) {
        Some(_) => 24,
        None => 24, //~ ERROR: this match arm has an identical body to another arm
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
        (None, Some(a)) => bar(a), //~ ERROR: this match arm has an identical body to another arm
        _ => (),
    }

    match (Some(42), Some(42)) {
        (Some(a), ..) => bar(a), //~ ERROR: this match arm has an identical body to another arm
        (.., Some(a)) => bar(a),
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
        (Ok(x), Some(_)) => println!("ok {}", x), //~ ERROR: this match arm has an identical body to another arm
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
        Ok(_) => println!("ok"), //~ ERROR: this match arm has an identical body to another arm
        Err(_) => {
            unreachable!();
        },
    }

    // False positive #1390
    macro_rules! empty {
        ($e:expr) => {};
    }
    match 0 {
        0 => {
            empty!(0);
        },
        1 => {
            empty!(1);
        },
        x => {
            empty!(x);
        },
    };

    // still lint if the tokens are the same
    match 0 {
        0 => {
            empty!(0);
        },
        1 => {
            //~^ ERROR: this match arm has an identical body to another arm
            empty!(0);
        },
        x => {
            empty!(x);
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

fn main() {
    let _ = match Some(0) {
        Some(0) => 0,
        Some(1) => 1,
        #[cfg(feature = "foo")]
        Some(2) => 2,
        _ => 1,
    };

    enum Foo {
        X(u32),
        Y(u32),
        Z(u32),
    }

    // Don't lint. `Foo::X(0)` and `Foo::Z(_)` overlap with the arm in between.
    let _ = match Foo::X(0) {
        Foo::X(0) => 1,
        Foo::X(_) | Foo::Y(_) | Foo::Z(0) => 2,
        Foo::Z(_) => 1,
        _ => 0,
    };

    // Suggest moving `Foo::Z(_)` up.
    let _ = match Foo::X(0) {
        Foo::X(0) => 1, //~ ERROR: this match arm has an identical body to another arm
        Foo::X(_) | Foo::Y(_) => 2,
        Foo::Z(_) => 1,
        _ => 0,
    };

    // Suggest moving `Foo::X(0)` down.
    let _ = match Foo::X(0) {
        Foo::X(0) => 1,
        Foo::Y(_) | Foo::Z(0) => 2,
        Foo::Z(_) => 1, //~ ERROR: this match arm has an identical body to another arm
        _ => 0,
    };

    // Don't lint.
    let _ = match 0 {
        -2 => 1,
        -5..=50 => 2,
        -150..=88 => 1,
        _ => 3,
    };

    struct Bar {
        x: u32,
        y: u32,
        z: u32,
    }

    // Lint.
    let _ = match None {
        Some(Bar { x: 0, y: 5, .. }) => 1,
        Some(Bar { y: 10, z: 0, .. }) => 2,
        None => 50,
        Some(Bar { y: 0, x: 5, .. }) => 1, //~ ERROR: this match arm has an identical body to another arm
        _ => 200,
    };

    let _ = match 0 {
        0 => todo!(),
        1 => todo!(),
        2 => core::convert::identity::<u32>(todo!()),
        3 => core::convert::identity::<u32>(todo!()),
        _ => 5,
    };

    let _ = match 0 {
        0 => cfg!(not_enable),
        1 => cfg!(not_enable),
        _ => false,
    };
}
