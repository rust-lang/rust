// Test calling methods on an impl for a bare trait. This test checks that the
// trait impl is only applied to a trait object, not concrete types which implement
// the trait.

trait T {}

impl<'a> dyn T + 'a {
    fn foo(&self) {}
}

impl T for i32 {}

fn main() {
    let x = &42i32;
    x.foo(); //~ERROR: no method named `foo` found for type `&i32` in the current scope
}
