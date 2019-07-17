#![feature(exclusive_range_pattern)]
#![warn(clippy::all)]
#![allow(unused, clippy::redundant_pattern_matching, clippy::too_many_lines)]
#![warn(clippy::match_same_arms)]

fn dummy() {}

fn match_wild_err_arm() {
    let x: Result<i32, &str> = Ok(3);

    match x {
        Ok(3) => println!("ok"),
        Ok(_) => println!("ok"),
        Err(_) => panic!("err"),
    }

    match x {
        Ok(3) => println!("ok"),
        Ok(_) => println!("ok"),
        Err(_) => panic!(),
    }

    match x {
        Ok(3) => println!("ok"),
        Ok(_) => println!("ok"),
        Err(_) => {
            panic!();
        },
    }

    // Allowed when not with `panic!` block.
    match x {
        Ok(3) => println!("ok"),
        Ok(_) => println!("ok"),
        Err(_) => println!("err"),
    }

    // Allowed when used with `unreachable!`.
    match x {
        Ok(3) => println!("ok"),
        Ok(_) => println!("ok"),
        Err(_) => unreachable!(),
    }

    match x {
        Ok(3) => println!("ok"),
        Ok(_) => println!("ok"),
        Err(_) => unreachable!(),
    }

    match x {
        Ok(3) => println!("ok"),
        Ok(_) => println!("ok"),
        Err(_) => {
            unreachable!();
        },
    }

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

    // Because of a bug, no warning was generated for this case before #2251.
    match x {
        Ok(_tmp) => println!("ok"),
        Ok(3) => println!("ok"),
        Ok(_) => println!("ok"),
        Err(_) => {
            unreachable!();
        },
    }
}

fn main() {}
