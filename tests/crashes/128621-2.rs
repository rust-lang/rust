//@ known-bug: rust-lang/rust#128621

#![feature(ptr_metadata)]
use std::{ops::FnMut, ptr::Pointee};

pub type EmplacerFn<'a, T> = dyn for<'b> FnMut(<T as Pointee>::Metadata) + 'a;

pub struct Emplacer<'a, T>(EmplacerFn<'a, T>);

impl<'a, T> Emplacer<'a, T> {
    pub unsafe fn from_fn<'b>(emplacer_fn: &'b mut EmplacerFn<'a, T>) -> &'b mut Self {
        unsafe { &mut *((emplacer_fn as *mut EmplacerFn<'a, T>) as *mut Self) }
    }
}

pub fn main() {}
