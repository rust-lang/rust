
use std::mem::ManuallyDrop;
use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut};

pub type MoveSlot<'a, T> = Move<'a, MaybeUninit<T>>;

pub struct Move<'a, T>(&'a mut ManuallyDrop<T>);

impl<'a, T> Move<'a, MaybeUninit<T>> {
    pub fn uninit(ptr: &'a mut ManuallyDrop<MaybeUninit<T>>) -> Self {
        Move(ptr)
    }

    // Assumes that MaybeUninit is #[repr(transparent)]
    pub fn init(&mut self, value: T) -> Move<'a, T> {
        *self.0 = ManuallyDrop::new(MaybeUninit::new(value));
        Move(unsafe { &mut *(self.0 as *mut _ as *mut ManuallyDrop<T>) })
    }
}

#[macro_export]
#[allow_internal_unstable]
macro_rules! uninit_slot {
    () => (&mut std::mem::ManuallyDrop::new(std::mem::MaybeUninit::uninitialized()))
}

impl<'a, T> Deref for Move<'a, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl<'a, T> DerefMut for Move<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.0
    }
}

impl<'a, T> Drop for Move<'a, T> {
    fn drop(&mut self) {
        unsafe {
            ManuallyDrop::drop(&mut self.0)
        }
    }
}
