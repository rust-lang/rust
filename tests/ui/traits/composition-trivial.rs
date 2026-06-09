//@ check-pass

trait Foo {
    fn foo(&self);
}

trait Bar : Foo {
    fn bar(&self);
}

pub fn main() {}
