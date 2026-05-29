// Regression test of #86162.
#![allow(todo_macro_calls)]

fn foo(x: impl Clone) {}
fn gen<T>() -> T { todo!() }

fn main() {
    foo(gen()); //<- Do not suggest `foo::<impl Clone>()`!
    //~^ ERROR: type annotations needed
}
