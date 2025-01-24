#![feature(ptr_metadata)]
// This test is the result of minimizing the `emplacable` crate to reproduce
// <https://github.com/rust-lang/miri/issues/3541>.

use std::ops::FnMut;
use std::ptr::Pointee;

pub type EmplacerFn<'a, T> = dyn for<'b> FnMut(<T as Pointee>::Metadata) + 'a;

#[repr(transparent)]
pub struct Emplacer<'a, T>(EmplacerFn<'a, T>)
where
    T: ?Sized;

impl<'a, T> Emplacer<'a, T>
where
    T: ?Sized,
{
    pub unsafe fn from_fn<'b>(emplacer_fn: &'b mut EmplacerFn<'a, T>) -> &'b mut Self {
        // This used to trigger:
        // constructing invalid value: wrong trait in wide pointer vtable: expected
        // `std::ops::FnMut(<[std::boxed::Box<i32>] as std::ptr::Pointee>::Metadata)`, but encountered
        // `std::ops::FnMut<(usize,)>`.
        unsafe { &mut *((emplacer_fn as *mut EmplacerFn<'a, T>) as *mut Self) }
    }
}

pub fn box_new_with<T>()
where
    T: ?Sized,
{
    let emplacer_closure = &mut |_meta| {
        unreachable!();
    };

    unsafe { Emplacer::<T>::from_fn(emplacer_closure) };
}

fn main() {
    box_new_with::<[Box<i32>]>();
}
