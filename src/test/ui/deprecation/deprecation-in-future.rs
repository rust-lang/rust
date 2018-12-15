// ignore-tidy-linelength

#![deny(deprecated_in_future)]

#[deprecated(since = "99.99.99", note = "text")]
pub fn deprecated_future() {}

fn test() {
    deprecated_future(); //~ ERROR use of item 'deprecated_future' that will be deprecated in future version 99.99.99: text
}

fn main() {}
