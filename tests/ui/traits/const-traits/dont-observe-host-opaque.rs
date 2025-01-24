//@ compile-flags: -Znext-solver
//@ check-pass

#![feature(const_trait_impl)]

const fn opaque() -> impl Sized {}

fn main() {
    let mut x = const { opaque() };
    x = opaque();
}
