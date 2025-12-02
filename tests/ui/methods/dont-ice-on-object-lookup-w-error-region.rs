// Fix for issue: #122914
#![allow(todo_macro_calls)]

use std::future::Future;
use std::pin::Pin;

fn project(x: Pin<&'missing mut dyn Future<Output = ()>>) {
    //~^ ERROR use of undeclared lifetime name `'missing`
    let _ = x.poll(todo!());
}

fn main() {}
