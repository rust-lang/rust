//@ check-pass
#![feature(staged_api)]
#![stable(feature = "test", since = "3.3.3")]

#[stable(feature = "test", since = "3.3.3")]
pub struct A<T>(pub T);

#[stable(feature = "test", since = "3.3.3")]
pub struct B<T>(#[stable(feature = "test", since = "3.3.3")] pub T);

fn main() {
    // Make sure the field is used to fill the stability cache
    A(0).0;
    B(0).0;
}
