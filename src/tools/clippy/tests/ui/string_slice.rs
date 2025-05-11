#![warn(clippy::string_slice)]
#![allow(clippy::no_effect)]

use std::borrow::Cow;

fn main() {
    &"Ölkanne"[1..];
    //~^ string_slice

    let m = "Mötörhead";
    &m[2..5];
    //~^ string_slice

    let s = String::from(m);
    &s[0..2];
    //~^ string_slice

    let a = Cow::Borrowed("foo");
    &a[0..3];
    //~^ string_slice
}
