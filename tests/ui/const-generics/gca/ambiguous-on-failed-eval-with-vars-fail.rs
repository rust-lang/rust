//@ revisions: old next
//@[next] compile-flags: -Znext-solver
// (`test_free_mismatch` is quite difficult to implement in the old solver, so make sure this test
// runs on the old solver, just in case someone attempts to implement GCA for the old solver and
// removes the restriction that -Znext-solver must be enabled)

#![feature(generic_const_items)]
#![feature(min_generic_const_args)]
#![feature(generic_const_args)]
//[old]~^ ERROR next-solver
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
    //[next]~^ ERROR type annotations needed
    arr_with_weird_len = [(); 10];
}

fn test_free_mismatch() {
    let (mut arr, mut arr_with_weird_len) = free();
    //[next]~^ ERROR type mismatch resolving `10 == 2`
    arr_with_weird_len = [(); 2];
    arr = [(); 10];
}

fn proj<const N: usize>() -> ([(); N], [(); <S as Trait>::PROJ::<N>]) {
    loop {}
}

fn test_proj() {
    let (mut arr, mut arr_with_weird_len) = proj();
    //[next]~^ ERROR type annotations needed
    arr_with_weird_len = [(); 10];
}

fn test_proj_mismatch() {
    let (mut arr, mut arr_with_weird_len) = proj();
    //[next]~^ ERROR type mismatch resolving `10 == 2`
    arr_with_weird_len = [(); 2];
    arr = [(); 10];
}

fn main() {
    test_free();
    test_proj();
}
