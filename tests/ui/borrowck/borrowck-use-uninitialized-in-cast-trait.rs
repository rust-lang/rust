// Variation on `borrowck-use-uninitialized-in-cast` in which we do a
// trait cast from an uninitialized source. Issue #20791.
//@ run-rustfix
#![allow(unused_variables, dead_code)]

trait Foo { fn dummy(&self) { } }
impl Foo for i32 { }

fn main() {
    let x: &i32;
    let y = x as *const dyn Foo; //~ ERROR [E0381]
}
