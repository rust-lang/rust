//@ run-pass
//@ run-rustfix

extern fn _foo() {}
//~^ WARN extern declarations without an explicit ABI are deprecated
//~^^ WARN this is accepted in the current edition (Rust 2015) but is a hard error in Rust future!

fn main() {}
