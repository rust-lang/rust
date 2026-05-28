// https://github.com/rust-lang/rust/issues/18685
//@ run-pass
// Test that the self param space is not used in a conflicting
// manner by unboxed closures within a default method on a trait

trait Tr {
    fn foo(&self);

    fn bar(&self) {
        (|| { self.foo() })()
    }
}

impl Tr for () {
    fn foo(&self) {}
}

fn main() {
    ().bar();
}
