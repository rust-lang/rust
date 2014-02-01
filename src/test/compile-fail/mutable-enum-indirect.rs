// Tests that an `&` pointer to something inherently mutable is itself
// to be considered mutable.

use std::kinds::marker;

enum Foo { A(marker::NoFreeze) }

fn bar<T: Freeze>(_: T) {}

fn main() {
    let x = A(marker::NoFreeze);
    bar(&x); //~ ERROR type parameter with an incompatible type
}
