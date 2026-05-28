//@ check-pass
//@ compile-flags: -Znext-solver

#![feature(generic_const_items)]
#![feature(min_generic_const_args)]
#![feature(generic_const_args)]
#![expect(incomplete_features)]

const FREE<const A: usize>: usize = 10;

trait Trait {
    const PROJ<const A: usize>: usize;
}
struct S;

impl Trait for S {
    const PROJ<const A: usize>: usize = 10;
}

fn free<const N: usize>() -> ([(); N], [(); FREE::<N>]) {
    loop {}
}

fn test_free() {
    let (mut arr, mut arr_with_weird_len) = free();
    arr_with_weird_len = [(); 10];
    arr = [(); 10];
}

fn proj<const N: usize>() -> ([(); N], [(); <S as Trait>::PROJ::<N>]) {
    loop {}
}

fn test_proj() {
    let (mut arr, mut arr_with_weird_len) = proj();
    arr_with_weird_len = [(); 10];
    arr = [(); 10];
}

fn main() {
    test_free();
    test_proj();
}
