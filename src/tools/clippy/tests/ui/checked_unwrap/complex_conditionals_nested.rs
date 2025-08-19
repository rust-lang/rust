#![deny(clippy::panicking_unwrap, clippy::unnecessary_unwrap)]
#![allow(
    clippy::if_same_then_else,
    clippy::branches_sharing_code,
    clippy::unnecessary_literal_unwrap
)]
//@no-rustfix: has placeholders
fn test_nested() {
    fn nested() {
        let x = Some(());
        if x.is_some() {
            // unnecessary
            x.unwrap();
            //~^ unnecessary_unwrap
        } else {
            // will panic
            x.unwrap();
            //~^ panicking_unwrap
        }
    }
}

fn main() {}
