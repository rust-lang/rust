//! Regression test for <https://github.com/rust-lang/rust/issues/125564>.

//@ incremental

#![allow(incomplete_features)]
#![feature(adt_const_params, unsized_const_params, generic_const_exprs)]

const fn concat_strs() -> &'static str {
    //~^ ERROR mismatched types
    const fn concat_arr<const M: usize, const N: usize>(a: [u8; M], b: [u8; N]) -> [u8; M + N] {}
    //~^ ERROR mismatched types

    impl<const A: &'static str, const B: &'static str> Inner<A, B>
    //~^ ERROR cannot find type `Inner` in this scope
    where
        [(); A.len()]:,
        [(); B.len()]:,
        [(); A.len() + B.len()]:,
    {
        const ABSTR: &'static str = unsafe {
            std::str::from_utf8_unchecked(&concat_arr(
                A.as_ptr().cast().read(),
                //~^ WARN type annotations needed
                //~| WARN this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2018!
                B.as_ptr().cast().read(),
                //~^ WARN type annotations needed
                //~| WARN this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2018!
            ))
        };
    }
}

fn main() {}
