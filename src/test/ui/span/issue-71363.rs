// compile-flags: -Z simulate-remapped-rust-src-base=/rustc/xyz -Z ui-testing=no
// only-x86_64-unknown-linux-gnu
//---^ Limiting target as the above unstable flags don't play well on some environment.

struct MyError;
impl std::error::Error for MyError {}
//~^ ERROR: `MyError` doesn't implement `std::fmt::Display`
//~| ERROR: `MyError` doesn't implement `Debug`

fn main() {}
