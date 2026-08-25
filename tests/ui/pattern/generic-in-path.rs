//! Ensure we can deal with a pattern that depends on a generic in its path, but the
//! actual pattern value can be computed independent of the generic.
#[derive(PartialEq)]
pub struct Thing<const N: usize>;

impl<const N: usize> Thing<N> {
    const A: Self = Thing;
}

fn broken<const N: usize>(x: Thing<N>) {
    match x {
        <Thing<N>>::A => {} //~ERROR: cannot depend on generic
        _ => {}
    }
}

fn main() {}
