//@ known-bug: #150296
#[derive(PartialEq)]
pub struct Thing<const N: usize>;

impl<const N: usize> Thing<N> {
    const A: Self = Thing;
}

fn broken<const N: usize>(x: Thing<N>) {
    match x {
        <Thing<N>>::A => {}
        _ => {}
    }
}
