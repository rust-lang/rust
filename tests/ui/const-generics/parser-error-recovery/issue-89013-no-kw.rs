trait Foo<const N: usize> {
    fn do_x(&self) -> [u8; N];
}

struct Bar;

const T: usize = 42;

impl Foo<N = 3> for Bar {
//~^ ERROR trait takes 1 generic argument but 0 generic arguments were supplied
//~| ERROR associated type bindings are not allowed here
//~| ERROR associated const equality is incomplete
    fn do_x(&self) -> [u8; 3] {
        [0u8; 3]
    }
}

fn main() {}
