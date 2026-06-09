// Duplicate implementations of Copy/Clone should not trigger
// borrow check warnings
// See #131083

#[derive(Copy, Clone)]
#[derive(Copy, Clone)]
//~^ ERROR conflicting implementations of trait `Clone` for type `E`
//~| ERROR conflicting implementations of trait `Copy` for type `E`
enum E {}

fn main() {}
