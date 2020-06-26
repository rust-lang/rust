#![deny(clippy::match_bool)]

fn match_bool() {
    let test: bool = true;

    match test {
        true => 0,
        false => 42,
    };

    let option = 1;
    match option == 1 {
        true => 1,
        false => 0,
    };

    match test {
        true => (),
        false => {
            println!("Noooo!");
        },
    };

    match test {
        false => {
            println!("Noooo!");
        },
        _ => (),
    };

    match test && test {
        false => {
            println!("Noooo!");
        },
        _ => (),
    };

    match test {
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
}

fn main() {}
