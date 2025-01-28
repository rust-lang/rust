#![deny(clippy::match_bool)]
#![allow(clippy::nonminimal_bool, clippy::eq_op)]

fn match_bool() {
    let test: bool = true;

    match test {
        //~^ ERROR: `match` on a boolean expression
        true => 0,
        false => 42,
    };

    let option = 1;
    match option == 1 {
        //~^ ERROR: `match` on a boolean expression
        true => 1,
        false => 0,
    };

    match test {
        //~^ ERROR: `match` on a boolean expression
        true => (),
        false => {
            println!("Noooo!");
        },
    };

    match test {
        //~^ ERROR: `match` on a boolean expression
        false => {
            println!("Noooo!");
        },
        _ => (),
    };

    match test && test {
        //~^ ERROR: `match` on a boolean expression
        false => {
            println!("Noooo!");
        },
        _ => (),
    };

    match test {
        //~^ ERROR: `match` on a boolean expression
        false => {
            println!("Noooo!");
        },
        true => {
            println!("Yes!");
        },
    };

    // Not linted
    match option {
        1..=10 => 1,
        11..=20 => 2,
        _ => 3,
    };

    // Don't lint
    let _ = match test {
        #[cfg(feature = "foo")]
        true if option == 5 => 10,
        true => 0,
        false => 1,
    };

    let _ = match test {
        //~^ ERROR: `match` on a boolean expression
        true if option == 5 => 10,
        _ => 1,
    };

    let _ = match test {
        //~^ ERROR: `match` on a boolean expression
        false if option == 5 => 10,
        _ => 1,
    };

    match test {
        //~^ ERROR: `match` on a boolean expression
        true if option == 5 => println!("Hello"),
        _ => (),
    };

    match test {
        //~^ ERROR: `match` on a boolean expression
        true if option == 5 => (),
        _ => println!("Hello"),
    };

    match test {
        //~^ ERROR: `match` on a boolean expression
        false if option == 5 => println!("Hello"),
        _ => (),
    };

    match test {
        //~^ ERROR: `match` on a boolean expression
        false if option == 5 => (),
        _ => println!("Hello"),
    };
}

fn main() {}
