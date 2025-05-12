//@ known-bug: #118952
#![allow(non_camel_case_types)]
#![feature(generic_const_exprs)]
#![feature(specialization)]

const DEFAULT_SMALL_VEC_INLINE_CAPACITY: usize = std::mem::size_of::<usize>() * 8;

pub const fn tiny_vec_cap<T>() -> usize {
    return (DEFAULT_SMALL_VEC_INLINE_CAPACITY - 1) / std::mem::size_of::<T>()
}

pub struct TinyVec<T, const N: usize = {tiny_vec_cap::<T>()}>
    where [
       ();
       (N * std::mem::size_of::<T>())
       - std::mem::size_of::<std::ptr::NonNull<T>>()
       - std::mem::size_of::<isize>()
    ]: ,
{
    data: isize //TinyVecData<T, N>,
}

pub fn main() {
    let t = TinyVec::<u8>::new();
}
