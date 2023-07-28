//@no-rustfix: overlapping suggestions
#![deny(clippy::match_bool)]

fn match_bool() {
    let test: bool = true;

    match test {
        //~^ ERROR: you seem to be trying to match on a boolean expression
        true => 0,
        false => 42,
    };

    let option = 1;
    match option == 1 {
        //~^ ERROR: you seem to be trying to match on a boolean expression
        true => 1,
        false => 0,
    };

    match test {
        //~^ ERROR: you seem to be trying to match on a boolean expression
        true => (),
        false => {
            println!("Noooo!");
        },
    };

    match test {
        //~^ ERROR: you seem to be trying to match on a boolean expression
        false => {
            println!("Noooo!");
        },
        _ => (),
    };

    match test && test {
        //~^ ERROR: this boolean expression can be simplified
        //~| NOTE: `-D clippy::nonminimal-bool` implied by `-D warnings`
        //~| ERROR: you seem to be trying to match on a boolean expression
        //~| ERROR: equal expressions as operands to `&&`
        //~| NOTE: `#[deny(clippy::eq_op)]` on by default
        false => {
            println!("Noooo!");
        },
        _ => (),
    };

    match test {
        //~^ ERROR: you seem to be trying to match on a boolean expression
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
}

fn main() {}
