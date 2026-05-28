// Regression test for #143523.

trait Trait {}

impl Trait for Vec<i32> {}

fn foo(_: impl Trait) {}

fn main() {
    foo(&mut vec![1]);
    //~^ ERROR the trait bound `&mut Vec<{integer}>: Trait` is not satisfied
}
