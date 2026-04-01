pub struct Foo<'a, const N: usize, T: 'a + ?Sized>(pub &'a T, [(); N]);
