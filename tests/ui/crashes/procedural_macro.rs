#[macro_use]
extern crate clippy_mini_macro_test;

#[deny(warnings)]
fn main() {
    let x = Foo;
    println!("{:?}", x);
}

#[derive(ClippyMiniMacroTest, Debug)]
struct Foo;
