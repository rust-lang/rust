// run-pass
// pretty-expanded FIXME #23616

trait Foo {
    fn foo(&self);
}

trait Bar : Foo {
    fn bar(&self);
}

pub fn main() {}
