//@ check-pass
#![allow(dead_code)]
trait Trait {
    type Output;
}

fn f<T: Trait>() {
    std::mem::size_of::<T::Output>();
}

fn main() {}
