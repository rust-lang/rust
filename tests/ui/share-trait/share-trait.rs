//@ check-pass

#![feature(share_trait)]

use std::clone::Share;

#[derive(Debug, PartialEq)]
struct Alias(u8);

impl Clone for Alias {
    fn clone(&self) -> Self {
        Alias(self.0 + 1)
    }
}

impl Share for Alias {}

fn share_generic<T: Share>(value: &T) -> T {
    value.share()
}

fn main() {
    let value = Alias(1);

    assert_eq!(Share::share(&value), Alias(2));
    assert_eq!(value.share(), Alias(2));
    assert_eq!(share_generic(&value), Alias(2));
}
