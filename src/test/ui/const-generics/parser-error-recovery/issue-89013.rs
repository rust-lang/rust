trait Foo<const N: usize> {
    fn do_x(&self) -> [u8; N];
}

struct Bar;

const T: usize = 42;

impl Foo<N = const 3> for Bar {
//~^ ERROR expected lifetime, type, or constant, found keyword `const`
//~| ERROR this trait takes 1 generic
//~| ERROR associated type bindings are not allowed here
//~| ERROR associated const equality is incomplete
    fn do_x(&self) -> [u8; 3] {
        [0u8; 3]
    }
}

fn main() {}
