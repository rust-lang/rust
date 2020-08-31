// run-pass

#![feature(cow_is_borrowed)]

use std::borrow::Cow;

fn main() {
    const COW: Cow<str> = Cow::Borrowed("moo");

    const IS_BORROWED: bool = COW.is_borrowed();
    assert!(IS_BORROWED);

    const IS_OWNED: bool = COW.is_owned();
    assert!(!IS_OWNED);
}
