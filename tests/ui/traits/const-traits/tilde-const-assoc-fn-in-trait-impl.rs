// Regression test for issue #119700.
//@ check-pass
//@ compile-flags: -Znext-solver
#![feature(const_trait_impl)]

#[const_trait]
trait Main {
    (const) fn compute<T: ~const Aux>() -> u32;
}

impl const Main for () {
    (const) fn compute<T: ~const Aux>() -> u32 {
        T::generate()
    }
}

#[const_trait]
trait Aux {
    (const) fn generate() -> u32;
}

impl const Aux for () {
    (const) fn generate() -> u32 { 1024 }
}

fn main() {
    const _: u32 = <()>::compute::<()>();
    let _ = <()>::compute::<()>();
}
