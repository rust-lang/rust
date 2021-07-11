// check-pass
#[doc(primitive = "usize")]
//~^ WARNING `doc(primitive)` should never have been stable
//~| WARNING hard error in a future release
/// Some docs
mod usize {}

fn main() {}
