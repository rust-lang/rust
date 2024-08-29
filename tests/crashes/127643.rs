//@ known-bug: #127643

#![feature(associated_const_equality)]

fn user() -> impl Owner<dyn Sized, C = 0> {}

trait Owner<K> {
    const C: K;
}
impl<K: ConstDefault> Owner<K> for () {
    const C: K = K::DEFAULT;
}

trait ConstDefault {
    const DEFAULT: Self;
}

fn main() {}
