// compile-pass

#![feature(custom_attribute)]

#[my_attr(a b c d)]
#[my_attr[a b c d]]
#[my_attr{a b c d}]
fn main() {}
