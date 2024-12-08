//@ check-pass

#![deny(deprecated_in_future)]

#[deprecated(since = "99.99.99", note = "text")]
pub fn deprecated_future() {}

fn test() {
    deprecated_future(); // ok; deprecated_in_future only applies with `#![feature(staged_api)]`
    //~^ WARNING use of deprecated function `deprecated_future`: text [deprecated]
}

fn main() {}
