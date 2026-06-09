// Tests that an `&` pointer to something inherently mutable is itself
// to be considered mutable.

#![feature(negative_impls)]

use std::marker::Sync;

struct NoSync;
impl !Sync for NoSync {}

enum Foo { A(NoSync) }

fn bar<T: Sync>(_: T) {}

fn main() {
    let x = Foo::A(NoSync);
    bar(&x);
    //~^ ERROR `NoSync` cannot be shared between threads safely [E0277]
}
