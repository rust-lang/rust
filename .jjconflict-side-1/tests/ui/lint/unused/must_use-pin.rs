#![deny(unused_must_use)]

use std::{ops::Deref, pin::Pin};

#[must_use]
struct MustUse;

#[must_use]
struct MustUsePtr<'a, T>(&'a T);

impl<'a, T> Deref for MustUsePtr<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

fn pin_ref() -> Pin<&'static ()> {
    Pin::new(&())
}

fn pin_ref_mut() -> Pin<&'static mut ()> {
    Pin::new(unimplemented!())
}

fn pin_must_use_ptr() -> Pin<MustUsePtr<'static, ()>> {
    Pin::new(MustUsePtr(&()))
}

fn pin_box() -> Pin<Box<()>> {
    Box::pin(())
}

fn pin_box_must_use() -> Pin<Box<MustUse>> {
    Box::pin(MustUse)
}

fn main() {
    pin_ref();
    pin_ref_mut();
    pin_must_use_ptr(); //~ ERROR unused pinned `MustUsePtr` that must be used
    pin_box();
    pin_box_must_use(); //~ ERROR unused pinned boxed `MustUse` that must be used
}
