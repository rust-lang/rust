// run-rustfix
#[allow(dead_code)]
struct AᐸTᐳ(T);
//~^ ERROR cannot find type `T` in this scope
//~| WARNING identifier contains uncommon Unicode codepoints
fn main() {}
