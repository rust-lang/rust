//@ known-bug: #118952
#![feature(generic_const_exprs)]

pub struct TinyVec<T, const N: usize = { () }>
where
    [(); () - std::mem::size_of() - std::mem::size_of::<isize>()]:, {}

pub fn main() {
    let t = TinyVec::<u8>::new();
}
