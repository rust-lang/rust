// check-pass
// edition:2018
// compile-flags: --crate-type lib

#![feature(return_position_impl_trait_v2)]

pub fn foo(f: impl Send) -> impl Send {
    f
}
