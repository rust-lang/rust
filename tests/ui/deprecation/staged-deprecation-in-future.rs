#![deny(deprecated_in_future)]

#![feature(staged_api)]

#![stable(feature = "rustc_deprecation_in_future_test", since = "1.0.0")]

#[deprecated(since = "99.99.99", note = "effectively never")]
#[stable(feature = "rustc_deprecation_in_future_test", since = "1.0.0")]
pub struct S1;

#[deprecated(since = "TBD", note = "literally never")]
#[stable(feature = "rustc_deprecation_in_future_test", since = "1.0.0")]
pub struct S2;

fn main() {
    let _ = S1; //~ ERROR use of unit struct `S1` that will be deprecated in future version 99.99.99: effectively never
    let _ = S2; //~ ERROR use of unit struct `S2` that will be deprecated in a future Rust version: literally never
}
