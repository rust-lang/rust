trait Bar<const N: usize> {}

trait Foo<const N: usize> {
    type Assoc: Bar<N>;
}

impl Bar<3> for u16 {}
impl<const N: usize> Foo<N> for i16 {
    type Assoc = u16; //~ ERROR trait `Bar<N>` is not implemented for `u16`
}

fn main() {}
