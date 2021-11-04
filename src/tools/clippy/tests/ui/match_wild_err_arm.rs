// revisions: edition2018 edition2021
// [edition2018] edition:2018
// [edition2021] edition:2021
#![feature(exclusive_range_pattern)]
#![allow(clippy::match_same_arms)]
#![warn(clippy::match_wild_err_arm)]

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

    match x {
        Ok(3) => println!("ok"),
        Ok(_) => println!("ok"),
        Err(_e) => panic!(),
    }

    // Allowed when used in `panic!`.
    match x {
        Ok(3) => println!("ok"),
        Ok(_) => println!("ok"),
        Err(_e) => panic!("{}", _e),
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

    // Allowed when used with `unreachable!`.
    match x {
        Ok(3) => println!("ok"),
        Ok(_) => println!("ok"),
        Err(_) => {
            unreachable!();
        },
    }
}

fn main() {}
