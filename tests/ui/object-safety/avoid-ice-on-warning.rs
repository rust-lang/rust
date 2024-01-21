fn call_this<F>(f: F) : Fn(&str) + call_that {}
//~^ ERROR return types are denoted using `->`
//~| ERROR cannot find trait `call_that` in this scope
//~| WARN trait objects without an explicit `dyn` are deprecated
//~| WARN this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!
fn main() {}
