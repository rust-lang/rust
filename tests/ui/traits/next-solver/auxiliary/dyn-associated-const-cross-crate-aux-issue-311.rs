use std::marker::PhantomData;
use std::mem::size_of;

pub trait Project {
    type Item;
}

impl<T: ?Sized> Project for T {
    type Item = u8;
}

pub type Matching = dyn Project<Item = u8>;
pub type Conflicting = dyn Project<Item = u16>;

pub struct Adapter<T: ?Sized>(PhantomData<T>);

impl<T: ?Sized + Project> Adapter<T> {
    pub const SIZE: usize = size_of::<T::Item>();
}

pub fn generic<T: ?Sized + Project>() -> usize {
    Adapter::<T>::SIZE
}

pub fn explicit_projection_size() -> usize {
    size_of::<<Matching as Project>::Item>()
}
