#![deny(clippy::match_bool)]
#![allow(clippy::nonminimal_bool, clippy::eq_op)]

fn match_bool() {
    let test: bool = true;

    match test {
        //~^ match_bool
        true => 0,
        false => 42,
    };

    let option = 1;
    match option == 1 {
        //~^ match_bool
        true => 1,
        false => 0,
    };

    match test {
        //~^ match_bool
        true => (),
        false => {
            println!("Noooo!");
        },
    };

    match test {
        //~^ match_bool
        false => {
            println!("Noooo!");
        },
        _ => (),
    };

    match test && test {
        //~^ match_bool
        false => {
            println!("Noooo!");
        },
        _ => (),
    };

    match test {
        //~^ match_bool
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
        //~^ match_bool
        true if option == 5 => 10,
        _ => 1,
    };

    let _ = match test {
        //~^ match_bool
        false if option == 5 => 10,
        _ => 1,
    };

    match test {
        //~^ match_bool
        true if option == 5 => println!("Hello"),
        _ => (),
    };

    match test {
        //~^ match_bool
        true if option == 5 => (),
        _ => println!("Hello"),
    };

    match test {
        //~^ match_bool
        false if option == 5 => println!("Hello"),
        _ => (),
    };

    match test {
        //~^ match_bool
        false if option == 5 => (),
        _ => println!("Hello"),
    };
}

fn issue14099() {
    match true {
        //~^ match_bool
        true => 'a: {
            break 'a;
        },
        _ => (),
    }
}

fn issue15351() {
    let mut d = false;
    match d {
        false => println!("foo"),
        ref mut d => *d = false,
    }

    match d {
        false => println!("foo"),
        e => println!("{e}"),
    }
}

fn main() {}
