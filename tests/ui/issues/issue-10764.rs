fn f(_: extern "Rust" fn()) {}
extern "C" fn bar() {}

fn main() { f(bar) }
//~^ ERROR mismatched types
