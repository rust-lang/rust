//@ check-pass

#![deny(keyword_idents_2024)]

macro_rules! foo {
    ($gen:expr) => {
        $gen
    };
}

fn main() {
    foo!(println!("hello, world"));
}
