#![feature(plugin)]
#![plugin(clippy)]

pub struct Lt<'a> {
    _foo: &'a u8,
}

impl<'a> Copy for Lt<'a> {}
impl<'a> Clone for Lt<'a> {
    fn clone(&self) -> Lt<'a> {
        unimplemented!();
    }
}

pub struct Ty<A> {
    _foo: A,
}

impl<A: Copy> Copy for Ty<A> {}
impl<A> Clone for Ty<A> {
    fn clone(&self) -> Ty<A> {
        unimplemented!();
    }
}
