// Check that we can define inherent methods on newtype enums that use
// an auto-ref'd receiver.

enum Foo = uint;

impl Foo {
    fn len(&self) -> uint { **self }
}

fn main() {
    let m = Foo(3);
    assert m.len() == 3;
}

