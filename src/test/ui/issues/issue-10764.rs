fn f(_: extern "Rust" fn()) {}
extern fn bar() {}

fn main() { f(bar) }
//~^ ERROR arguments to this function are incorrect
