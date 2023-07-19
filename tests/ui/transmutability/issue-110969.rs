// check-fail
// revisions: current next
//[next] compile-flags: -Ztrait-solver=next
#![feature(adt_const_params, generic_const_exprs, transmutability)]
#![allow(incomplete_features)]

mod assert {
    use std::mem::BikeshedIntrinsicFrom;

    pub fn is_transmutable<Src, Dst, Context, const ASSUME: std::mem::Assume>()
    where
        Dst: BikeshedIntrinsicFrom<Src, Context, ASSUME>,
    {
    }
}

fn main() {
    struct Context;
    #[repr(C)]
    struct Src;
    #[repr(C)]
    struct Dst;

    trait Trait {
        // The `{}` should not cause an ICE
        const FALSE: bool = assert::is_transmutable::<Src, Dst, Context, {}>();
        //~^ ERROR mismatched types
        //~^^ ERROR mismatched types
    }
}
