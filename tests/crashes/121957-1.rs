//@ known-bug: #121957
#![feature(const_trait_impl, effects)]

#[const_trait]
trait Main {
    fn compute<T: ~const Aux>() -> u32;
}

impl const Main for () {
    fn compute<'x, 'y, 'z: 'x>() -> u32 {}
}

#[const_trait]
trait Aux {}

impl const Aux for () {}

fn main() {
    const _: u32 = <()>::compute::<()>();
}
