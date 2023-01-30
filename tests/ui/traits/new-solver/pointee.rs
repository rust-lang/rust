// compile-flags: -Ztrait-solver=next
// check-pass
#![feature(ptr_metadata)]

use std::ptr::{DynMetadata, Pointee};

trait Trait<U> {}
struct MyDst<T: ?Sized>(T);

fn works<T>() {
    let _: <T as Pointee>::Metadata = ();
    let _: <[T] as Pointee>::Metadata = 1_usize;
    let _: <str as Pointee>::Metadata = 1_usize;
    let _: <dyn Trait<T> as Pointee>::Metadata = give::<DynMetadata<dyn Trait<T>>>();
    let _: <MyDst<T> as Pointee>::Metadata = ();
    let _: <((((([u8],),),),),) as Pointee>::Metadata = 1_usize;
}

fn give<U>() -> U {
    loop {}
}

fn main() {}
