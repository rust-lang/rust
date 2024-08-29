//@ check-pass
//@ compile-flags: -Znext-solver
// gate-test-effects
// ^ effects doesn't have a gate so we will trick tidy into thinking this is a gate test
#![allow(incomplete_features)]
#![feature(
    const_trait_impl,
    effects,
    core_intrinsics,
    const_eval_select
)]

// ensure we are passing in the correct host effect in always const contexts.

pub const fn hmm<T>() -> usize {
    // FIXME(const_trait_impl): maybe we should have a way to refer to the (hidden) effect param
    fn one() -> usize { 1 }
    const fn zero() -> usize { 0 }
    unsafe {
        std::intrinsics::const_eval_select((), zero, one)
    }
}

const _: () = {
    let x = hmm::<()>();
    assert!(0 == x);
};

pub const fn uwu(x: [u8; hmm::<()>()]) {
    let [] = x;
}

fn main() {}
