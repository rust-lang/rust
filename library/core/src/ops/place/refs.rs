use crate::marker::PhantomData;
use crate::ops::place::locals::LocalHandle;
use crate::ops::place::{DerefPlace, PlaceProxy};

pub struct RefHandle<'a, T> {
    ptr: *const T,
    _lt: PhantomData<&'a T>,
}

impl<'a, T> PlaceHandle for RefHandle<'a, T> {}

impl<'a, T> PlaceProxy for &'a T {
    type Handle = RefHandle<'a, T>;
}

impl<'a, T> DerefPlace for LocalHandle<&'a T> {
    unsafe fn deref_place(self) -> <Self::Target as PlaceProxy>::Handle {
        let ptr: *const &'a T = self.as_ptr();
        let ptr: *const *const T = ptr.cast();
        RefHandle { ptr: unsafe { *ptr }, _lt: PhantomData }
    }
}
