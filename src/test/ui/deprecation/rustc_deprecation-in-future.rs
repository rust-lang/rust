// ignore-tidy-linelength

#![deny(deprecated_in_future)]

#![feature(staged_api)]

#![stable(feature = "rustc_deprecation-in-future-test", since = "1.0.0")]

#[rustc_deprecated(since = "99.99.99", reason = "effectively never")]
#[stable(feature = "rustc_deprecation-in-future-test", since = "1.0.0")]
pub struct S1;

#[rustc_deprecated(since = "TBD", reason = "literally never")]
#[stable(feature = "rustc_deprecation-in-future-test", since = "1.0.0")]
pub struct S2;

fn main() {
    let _ = S1; //~ ERROR use of unit struct `S1` that will be deprecated in future version 99.99.99: effectively never
    let _ = S2; //~ ERROR use of unit struct `S2` that will be deprecated in a future Rust version: literally never
}
