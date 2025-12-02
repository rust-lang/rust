//@ compile-flags: -Znext-solver
#![allow(todo_macro_calls)]

trait Foo: Bar<i32> + Bar<u32> {}

trait Bar<T> {}

fn main() {
    let x: &dyn Foo = todo!();
    let y: &dyn Bar<usize> = x;
    //~^ ERROR mismatched types
}
