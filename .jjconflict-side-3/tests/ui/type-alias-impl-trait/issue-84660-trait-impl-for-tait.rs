// Regression test for issues #84660 and #86411: both are variations on #76202.
// Tests that we don't ICE when we have an opaque type appearing anywhere in an impl header.

//@ check-pass

#![feature(type_alias_impl_trait)]

trait Foo {}
impl Foo for () {}
type Bar = impl Foo;
#[define_opaque(Bar)]
fn _defining_use() -> Bar {}

trait TraitArg<T> {
    fn f();
}

impl TraitArg<Bar> for () {
    fn f() {
        println!("ho");
    }
}

fn main() {
    <() as TraitArg<Bar>>::f();
}
