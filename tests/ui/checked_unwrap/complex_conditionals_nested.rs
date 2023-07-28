#![deny(clippy::panicking_unwrap, clippy::unnecessary_unwrap)]
#![allow(
    clippy::if_same_then_else,
    clippy::branches_sharing_code,
    clippy::unnecessary_literal_unwrap
)]
//@no-rustfix
fn test_nested() {
    fn nested() {
        let x = Some(());
        if x.is_some() {
            // unnecessary
            x.unwrap();
        } else {
            // will panic
            x.unwrap();
        }
    }
}

fn main() {}
