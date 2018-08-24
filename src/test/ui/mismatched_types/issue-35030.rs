// rustc-env:RUST_NEW_ERROR_FORMAT

trait Parser<T> {
    fn parse(text: &str) -> Option<T>;
}

impl<bool> Parser<bool> for bool {
    fn parse(text: &str) -> Option<bool> {
        Some(true) //~ ERROR mismatched types
    }
}

fn main() {
    println!("{}", bool::parse("ok").unwrap_or(false));
}
