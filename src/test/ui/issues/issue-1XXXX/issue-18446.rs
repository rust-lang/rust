// Test that name clashes between the method in an impl for the type
// and the method in the trait when both are in the same scope.

trait T {
    fn foo(&self);
}

impl<'a> dyn T + 'a {
    fn foo(&self) {}
}

impl T for i32 {
    fn foo(&self) {}
}

fn main() {
    let x: &dyn T = &0i32;
    x.foo(); //~ ERROR multiple applicable items in scope [E0034]
}
