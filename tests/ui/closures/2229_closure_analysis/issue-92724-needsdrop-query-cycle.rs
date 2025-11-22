// ICEs if checking if there is a significant destructor causes a query cycle
//@ check-pass
#![allow(unconstructable_pub_struct)]

#![warn(rust_2021_incompatible_closure_captures)]
pub struct Foo(Bar);
pub struct Bar(Baz);
pub struct Baz(Vec<Foo>);

impl Foo {
    pub fn baz(self, v: Baz) -> Baz {
        (|| v)()
    }
}
fn main() {}
