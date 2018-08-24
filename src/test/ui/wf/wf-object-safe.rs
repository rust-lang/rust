// Check that object-safe traits are not WF when used as object types.
// Issue #21953.

trait A {
    fn foo(&self, _x: &Self);
}

fn main() {
    let _x: &A; //~ ERROR E0038
}
