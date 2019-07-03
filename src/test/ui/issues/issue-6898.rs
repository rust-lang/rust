// build-pass (FIXME(62277): could be check-pass?)
// pretty-expanded FIXME #23616

use std::mem;

/// Returns the size of a type
pub fn size_of<T>() -> usize {
    TypeInfo::size_of(None::<T>)
}

/// Returns the size of the type that `val` points to
pub fn size_of_val<T>(val: &T) -> usize {
    val.size_of_val()
}

pub trait TypeInfo: Sized {
    fn size_of(_lame_type_hint: Option<Self>) -> usize;
    fn size_of_val(&self) -> usize;
}

impl<T> TypeInfo for T {
    /// The size of the type in bytes.
    fn size_of(_lame_type_hint: Option<T>) -> usize {
        mem::size_of::<T>()
    }

    /// Returns the size of the type of `self` in bytes.
    fn size_of_val(&self) -> usize {
        TypeInfo::size_of(None::<T>)
    }
}

pub fn main() {}
