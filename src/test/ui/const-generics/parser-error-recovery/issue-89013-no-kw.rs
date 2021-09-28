trait Foo<const N: usize> {
    fn do_x(&self) -> [u8; N];
}

struct Bar;

const T: usize = 42;

impl Foo<N = 3> for Bar {
//~^ ERROR cannot constrain an associated constant to a value
//~| ERROR associated type bindings are not allowed here
    fn do_x(&self) -> [u8; 3] {
        [0u8; 3]
    }
}

fn main() {}
