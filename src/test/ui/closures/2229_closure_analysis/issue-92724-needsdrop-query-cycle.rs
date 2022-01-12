// ICEs if checking if there is a significant destructor causes a query cycle
// check-pass

#![warn(rust_2021_incompatible_closure_captures)]
pub struct Foo(Bar);
pub struct Bar(Vec<Foo>);

impl Foo {
    pub fn bar(self, v: Bar) -> Bar {
        (|| v)()
    }
}
fn main() {}
