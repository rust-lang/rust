// Regression test for https://github.com/rust-lang/rust/issues/141738
//
// Using a struct constructor as an array repeat count with
// `min_generic_const_args` used to ICE with "unexpected `DefKind`
// for const alias to resolve to: Ctor(Struct, Const)".
// It should now produce a proper type error.

#![feature(min_generic_const_args)]
//~^ WARN the feature `min_generic_const_args` is incomplete

struct S;

fn main() {
    let _b = [0; S];
    //~^ ERROR the constant `S` is not of type `usize`
}
