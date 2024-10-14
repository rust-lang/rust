//@ known-bug: #131101
trait Foo<const N: u8> {
    fn do_x(&self) -> [u8; N];
}

struct Bar;

impl Foo<const 3> for Bar {
    fn do_x(&self) -> [u8; 3] {
        [0u8; 3]
    }
}
