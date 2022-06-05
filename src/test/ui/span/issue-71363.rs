// ignore-test
// FIXME(#97682):
// The `-Z simulate-remapped-rust-src-base=/rustc/xyz -Z ui-testing=no` flags
// don't work well and make UI test fail on some env.
// Once it starts to work fine, we could re-enable them here.

struct MyError;
impl std::error::Error for MyError {}
//~^ ERROR: `MyError` doesn't implement `std::fmt::Display`
//~| ERROR: `MyError` doesn't implement `Debug`

fn main() {}
