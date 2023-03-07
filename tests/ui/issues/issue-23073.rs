#![feature(associated_type_defaults)]

trait Foo { type T; }
trait Bar {
    type Foo: Foo;
    type FooT = <<Self as Bar>::Foo>::T; //~ ERROR ambiguous associated type
}

fn main() {}
