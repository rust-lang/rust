// Regression test for #152694
// Verify that we don't emit invalid suggestions (adding `ref` or `clone()`)
// when a destructuring assignment involves a type that implements `Drop`.

struct Thing(String);

impl Drop for Thing {
    fn drop(&mut self) {}
}

fn main() {
    Thing(*&mut String::new()) = Thing(String::new());
    //~^ ERROR cannot move out of type `Thing`, which implements the `Drop` trait
}
