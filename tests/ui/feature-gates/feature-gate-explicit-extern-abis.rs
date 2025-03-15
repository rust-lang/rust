//@ run-pass
//@ run-rustfix

extern fn _foo() {}
//~^ WARN extern declarations without an explicit ABI are deprecated

fn main() {}
