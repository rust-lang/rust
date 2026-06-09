trait Foo<const N: usize> {
    fn do_x(&self) -> [u8; N];
}

struct Bar;

const T: usize = 42;

impl Foo<N = type 3> for Bar {
//~^ERROR missing type to the right of `=`
    fn do_x(&self) -> [u8; 3] {
        [0u8; 3]
    }
}

fn main() {}
