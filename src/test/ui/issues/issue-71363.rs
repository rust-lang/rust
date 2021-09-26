// check-fail
// compile-flags: -Z simulate-remapped-rust-src-base=/rustc/xyz -Z ui-testing=no

#[derive(Debug)]
struct Test;
impl std::error::Error for Test {}
//~^ ERROR `Test` doesn't implement `std::fmt::Display`

fn main() {}
