//! This test verifies that implementing a trait method with a signature that does not
//! exactly match its declaration in the trait results in a compilation error.
//! Specifically, it checks for errors when the number of parameters or the return type
//! in the `impl` differs from the trait definition.

trait Foo {
    fn foo(&mut self, x: i32, y: i32) -> i32;
}

impl Foo for i32 {
    fn foo(
        &mut self, //~ ERROR method `foo` has 2 parameters but the declaration
        x: i32,
    ) {
    }
}

fn main() {}
