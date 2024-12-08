//@ run-pass
// Test that subtyping the body of a static doesn't cause an ICE.

fn foo(_ : &()) {}
static X: fn(&'static ()) = foo;

fn main() {
    let _ = X;
}
