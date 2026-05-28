// #83056 ICE "bad input type for cast"
// issue: rust-lang/rust#83056

struct S([bool; f as usize]);
fn f() -> T {}
//~^ ERROR cannot find type `T` in this scope
pub fn main() {}
