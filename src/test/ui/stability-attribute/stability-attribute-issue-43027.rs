// check-pass
#![feature(staged_api)]
#![stable(feature = "test", since = "0")]

#[stable(feature = "test", since = "0")]
pub struct Reverse<T>(pub T); // if the field has no stability, we check its parent

fn main() {
    // Make sure the field is used to fill the stability cache
    Reverse(0).0;
}
