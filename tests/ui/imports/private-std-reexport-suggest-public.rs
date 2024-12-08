//@ run-rustfix
#![allow(unused_imports)]
fn main() {
    use foo::mem; //~ ERROR module import `mem` is private
}

pub mod foo {
    use std::mem;
}
