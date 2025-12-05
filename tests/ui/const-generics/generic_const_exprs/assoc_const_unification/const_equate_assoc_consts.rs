//@ check-pass
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

trait Trait {
    const ASSOC: usize;
}
impl<T> Trait for T {
    const ASSOC: usize = std::mem::size_of::<T>();
}

struct Foo<T: Trait>([u8; T::ASSOC])
where
    [(); T::ASSOC]:;

fn bar<T: Trait>()
where
    [(); T::ASSOC]:,
{
    let _: Foo<T> = Foo::<_>(make());
}

fn make() -> ! {
    todo!()
}

fn main() {}
