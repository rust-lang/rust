#![feature(optin_builtin_traits)]

use std::marker::Send;

struct NoSend;
impl !Send for NoSend {}

enum Foo {
    A(NoSend)
}

fn bar<T: Send>(_: T) {}

fn main() {
    let x = Foo::A(NoSend);
    bar(x);
    //~^ ERROR `NoSend` cannot be sent between threads safely
}
