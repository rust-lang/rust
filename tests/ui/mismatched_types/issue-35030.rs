#![allow(non_camel_case_types)]

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
