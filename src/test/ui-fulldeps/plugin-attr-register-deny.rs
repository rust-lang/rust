// aux-build:attr-plugin-test.rs
// ignore-stage1

#![feature(plugin)]
#![plugin(attr_plugin_test)]
#![deny(unused_attributes)]

#[baz]
fn baz() { } // no error

#[foo]
pub fn main() {
     //~^^ ERROR unused
    #[bar]
    fn inner() {}
    //~^^ ERROR crate
    //~^^^ ERROR unused
    baz();
    inner();
}
