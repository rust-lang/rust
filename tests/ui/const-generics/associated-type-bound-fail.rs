trait Bar<const N: usize> {}

trait Foo<const N: usize> {
    type Assoc: Bar<N>;
}

impl Bar<3> for u16 {}
impl<const N: usize> Foo<N> for i16 {
    type Assoc = u16; //~ ERROR the trait bound `u16: Bar<N>`
}

fn main() {}
