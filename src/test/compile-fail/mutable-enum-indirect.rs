// Tests that an `&` pointer to something inherently mutable is itself
// to be considered mutable.

use std::kinds::NotFreeze;

enum Foo { A(NotFreeze) }

fn bar<T: Freeze>(_: T) {}

fn main() {
    let x = A(NotFreeze);
    bar(&x); //~ ERROR type parameter with an incompatible type
}
