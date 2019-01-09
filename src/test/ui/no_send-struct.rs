#![feature(optin_builtin_traits)]

use std::marker::Send;

struct Foo {
    a: isize,
}

impl !Send for Foo {}

fn bar<T: Send>(_: T) {}

fn main() {
    let x = Foo { a: 5 };
    bar(x);
    //~^ ERROR `Foo` cannot be sent between threads safely
}
