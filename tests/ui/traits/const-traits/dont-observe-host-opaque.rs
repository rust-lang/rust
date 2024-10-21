//@ compile-flags: -Znext-solver
//@ check-pass

#![feature(const_trait_impl, effects)]
//~^ WARN the feature `effects` is incomplete

const fn opaque() -> impl Sized {}

fn main() {
    let mut x = const { opaque() };
    x = opaque();
}
