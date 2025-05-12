//@ run-pass

#![allow(unused_variables)]

pub trait TryTransform {
    fn try_transform<F>(self, f: F)
    where
        Self: Sized,
        F: FnOnce(Self);
}

impl<'a, T> TryTransform for &'a mut T {
    fn try_transform<F>(self, f: F)
    where
        // The bug was that `Self: Sized` caused the lifetime of `this` to "extend" for all
        // of 'a instead of only lasting as long as the binding is used (for just that line).
        Self: Sized,
        F: FnOnce(Self),
    {
        let this: *mut T = self as *mut T;
        f(self);
    }
}

fn main() {
}
