//@ check-pass

#![allow(clippy::let_and_return)]

fn main() {
    #[clippy::author]
    let a = match 42 {
        16 => 5,
        17 => {
            let x = 3;
            x
        },
        _ => 1,
    };
}
