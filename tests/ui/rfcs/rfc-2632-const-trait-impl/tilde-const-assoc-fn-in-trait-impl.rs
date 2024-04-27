// Regression test for issue #119700.
//@ check-pass

#![feature(const_trait_impl, effects)]

#[const_trait]
trait Main {
    fn compute<T: ~const Aux>() -> u32;
}

impl const Main for () {
    fn compute<T: ~const Aux>() -> u32 {
        T::generate()
    }
}

#[const_trait]
trait Aux {
    fn generate() -> u32;
}

impl const Aux for () {
    fn generate() -> u32 { 1024 }
}

fn main() {
    const _: u32 = <()>::compute::<()>();
    let _ = <()>::compute::<()>();
}
