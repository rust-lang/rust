#![feature(exclusive_range_pattern)]
#![allow(clippy::match_same_arms, dead_code)]
#![warn(clippy::match_wild_err_arm)]

fn issue_10635() {
    enum Error {
        A,
        B,
    }

    // Don't trigger in const contexts. Const unwrap is not yet stable
    const X: () = match Ok::<_, Error>(()) {
        Ok(x) => x,
        Err(_) => panic!(),
    };
}

fn match_wild_err_arm() {
    let x: Result<i32, &str> = Ok(3);

    match x {
        Ok(3) => println!("ok"),
        Ok(_) => println!("ok"),
        Err(_) => panic!("err"),
        //~^ ERROR: `Err(_)` matches all errors
        //~| NOTE: match each error separately or use the error output, or use `.expect(ms
    }

    match x {
        Ok(3) => println!("ok"),
        Ok(_) => println!("ok"),
        Err(_) => panic!(),
        //~^ ERROR: `Err(_)` matches all errors
        //~| NOTE: match each error separately or use the error output, or use `.expect(ms
    }

    match x {
        Ok(3) => println!("ok"),
        Ok(_) => println!("ok"),
        Err(_) => {
            //~^ ERROR: `Err(_)` matches all errors
            //~| NOTE: match each error separately or use the error output, or use `.expect(ms
            panic!();
        },
    }

    match x {
        Ok(3) => println!("ok"),
        Ok(_) => println!("ok"),
        Err(_e) => panic!(),
        //~^ ERROR: `Err(_e)` matches all errors
        //~| NOTE: match each error separately or use the error output, or use `.expect(ms
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
