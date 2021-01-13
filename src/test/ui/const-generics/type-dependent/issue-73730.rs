// check-pass
// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

trait Foo<'a, A>: Iterator<Item=A> {
    fn bar<const N: usize>(&mut self) -> *const [A; N];
}

impl<'a, A, I: ?Sized> Foo<'a, A> for I where I: Iterator<Item=A>  {
    fn bar<const N: usize>(&mut self) -> *const [A; N] {
        std::ptr::null()
    }
}

fn main() {
    (0_u8 .. 10).bar::<10_usize>();
}
