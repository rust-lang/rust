// compile-flags: -Cmetadata=aux

pub trait Deref {
    type Target: ?Sized;

    fn deref<'a>(&'a self) -> &'a Self::Target;
}

pub trait Add<RHS = Self> {
    type Output;

    fn add(self, rhs: RHS) -> Self::Output;
}


pub trait Bar {}
pub trait Deref2 {
    type Target: Bar;

    fn deref(&self) -> Self::Target;
}

pub trait Index<Idx: ?Sized> {
    type Output: ?Sized;
    fn index(&self, index: Idx) -> &Self::Output;
}

pub trait IndexMut<Idx: ?Sized>: Index<Idx> {
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output;
}
