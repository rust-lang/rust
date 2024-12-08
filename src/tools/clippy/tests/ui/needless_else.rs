#![allow(unused)]
#![warn(clippy::needless_else)]
#![allow(clippy::suspicious_else_formatting)]

macro_rules! mac {
    ($test:expr) => {
        if $test {
            println!("Test successful!");
        } else {
        }
    };
}

macro_rules! empty_expansion {
    () => {};
}

fn main() {
    let b = std::hint::black_box(true);

    if b {
        println!("Foobar");
    } else {
    }

    if b {
        println!("Foobar");
    } else {
        // Do not lint because this comment might be important
    }

    if b {
        println!("Foobar");
    } else
    /* Do not lint because this comment might be important */
    {
    }

    // Do not lint because of the expression
    let _ = if b { 1 } else { 2 };

    // Do not lint because inside a macro
    mac!(b);

    if b {
        println!("Foobar");
    } else {
        #[cfg(foo)]
        "Do not lint cfg'd out code"
    }

    if b {
        println!("Foobar");
    } else {
        empty_expansion!();
    }
}
