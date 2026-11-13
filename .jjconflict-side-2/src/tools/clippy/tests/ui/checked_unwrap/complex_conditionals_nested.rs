//@no-rustfix: has placeholders
#![warn(clippy::panicking_unwrap, clippy::unnecessary_unwrap)]
#![expect(clippy::branches_sharing_code, clippy::unnecessary_literal_unwrap)]

fn test_nested() {
    fn nested() {
        let x = Some(());
        if x.is_some() {
            x.unwrap();
            //~^ unnecessary_unwrap
        } else {
            x.unwrap();
            //~^ panicking_unwrap
        }
    }
}

fn main() {}
