#![deny(clippy::panicking_unwrap, clippy::unnecessary_unwrap)]
#![allow(
    clippy::if_same_then_else,
    clippy::branches_sharing_code,
    clippy::unnecessary_literal_unwrap
)]

fn test_nested() {
    fn nested() {
        let x = Some(());
        if x.is_some() {
            x.unwrap(); // unnecessary
        } else {
            x.unwrap(); // will panic
        }
    }
}

fn main() {}
