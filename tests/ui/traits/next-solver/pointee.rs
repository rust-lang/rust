//@ compile-flags: -Znext-solver
//@ check-pass
#![feature(ptr_metadata)]

use std::ptr::{DynMetadata, Pointee};

trait Trait<U> {}
struct MyDst<T: ?Sized>(T);

fn meta_is<T: Pointee<Metadata = U> + ?Sized, U>() {}

fn works<T>() {
    meta_is::<T, ()>();
    meta_is::<[T], usize>();
    meta_is::<str, usize>();
    meta_is::<dyn Trait<T>, DynMetadata<dyn Trait<T>>>();
    meta_is::<MyDst<T>, ()>();
    meta_is::<((((([u8],),),),),), usize>();
}

fn main() {}
