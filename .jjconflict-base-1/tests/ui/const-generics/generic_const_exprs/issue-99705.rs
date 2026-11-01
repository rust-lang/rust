//@ check-pass
#![crate_type = "lib"]
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
pub trait MyIterator {
    type Output;
}

pub trait Foo {
    const ABC: usize;
}

pub struct IteratorStruct<const N: usize>{

}

pub struct Bar<const N: usize> {
    pub data: [usize; N]
}

impl<const N: usize> MyIterator for IteratorStruct<N> {
    type Output = Bar<N>;
}

pub fn test1<T: Foo>() -> impl MyIterator<Output = Bar<{T::ABC}>> where [(); T::ABC]: Sized {
    IteratorStruct::<{T::ABC}>{}
}

pub trait Baz<const N: usize>{}
impl<const N: usize> Baz<N> for Bar<N> {}
pub fn test2<T: Foo>() -> impl MyIterator<Output = impl Baz<{ T::ABC }>> where [(); T::ABC]: Sized {
    IteratorStruct::<{T::ABC}>{}
}
