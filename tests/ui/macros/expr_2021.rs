//@ check-pass
//@ edition: 2015

// Ensures expr_2021 fragment specifier is accepted in old editions

macro_rules! my_macro {
    ($x:expr_2021) => {
        println!("Hello, {}!", $x);
    };
}

fn main() {
    my_macro!("world");
}
