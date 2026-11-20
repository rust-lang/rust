//@ edition:2021
// Regression test for #125805.
// ICE when using `offset_of!` on a field whose type is a bare trait object
// with a missing lifetime parameter.

trait X<'a> {}

use std::mem::offset_of;

struct T {
    y: X,
    //~^ ERROR missing lifetime specifier [E0106]
    //~| ERROR expected a type, found a trait [E0782]
}

fn other() {
    offset_of!(T, y);
}

fn main() {}
