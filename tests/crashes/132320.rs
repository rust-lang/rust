//@ known-bug: #132320
//@ compile-flags: -Znext-solver=globally

trait Foo {
    type Item;
    fn foo(&mut self);
}

impl Foo for () {
    type Item = Option<()>;

    fn foo(&mut self) {
        let _ = Self::Item::None;
    }
}
