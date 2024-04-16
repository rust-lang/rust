//@ known-bug: #121574
#![feature(generic_const_exprs)]

impl<const Z: usize> X<Z> {
    pub fn y<'a, U: 'a>(&'a self) -> impl Iterator<Item = impl Iterator<Item = [u8; Z]> + '_> {}
}
