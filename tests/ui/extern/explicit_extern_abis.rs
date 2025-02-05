//@ compile-flags: -Z unstable-options --edition 2027

extern fn foo() {}
//~^ ERROR extern declarations without an explicit ABI are disallowed

fn main() {}
