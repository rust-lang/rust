// known-bug
// compile-flags: --edition=2021 --crate-type=lib
// rustc-env:RUST_BACKTRACE=0

// tracked in https://github.com/rust-lang/rust/issues/96572

#![feature(type_alias_impl_trait)]

fn main() {
    type T = impl Copy;  // error: unconstrained opaque type
    let foo: T = (1u32, 2u32);
    let (a, b) = foo; // removing this line makes the code compile
}
