#![feature(custom_attribute)]

#[my_attr = !] //~ ERROR unexpected token: `!`
fn main() {}
