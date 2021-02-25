#![deny(clippy::panicking_unwrap, clippy::unnecessary_unwrap)]
#![allow(clippy::if_same_then_else, clippy::shared_code_in_if_blocks)]

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
