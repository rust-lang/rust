//! Regression test for <https://github.com/rust-lang/rust/issues/158656>.

//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ check-pass

trait Trait {}

trait WithAssoc {
    type Assoc: ?Sized;
}
struct Thing;
impl WithAssoc for Thing {
    type Assoc = dyn Trait;
}

fn foo() -> impl WithAssoc<Assoc = impl Trait + ?Sized> {
    Thing
}

fn main() {}
