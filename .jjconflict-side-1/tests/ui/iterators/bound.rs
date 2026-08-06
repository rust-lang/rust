struct S<I: Iterator>(I);
struct T(S<u8>);
//~^ ERROR is not an iterator
fn main() {}
