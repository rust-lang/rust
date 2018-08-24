#![feature(optin_builtin_traits)]

use std::marker::Sync;

struct Foo { a: isize }
impl !Sync for Foo {}

fn bar<T: Sync>(_: T) {}

fn main() {
    let x = Foo { a: 5 };
    bar(x);
    //~^ ERROR `Foo` cannot be shared between threads safely [E0277]
}
