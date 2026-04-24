use std::{mem, ptr};

use smallvec::SmallVec;
use thin_vec::ThinVec;

pub trait FlatMapInPlace<T> {
    /// `f` turns each element into 0..many elements. This function will consume the existing
    /// elements in a vec-like structure and replace them with any number of new elements — fewer,
    /// more, or the same number — as efficiently as possible.
    fn flat_map_in_place<F, I>(&mut self, f: F)
    where
        F: FnMut(T) -> I,
        I: IntoIterator<Item = T>;
}

// Blanket impl for all vec-like types that impl `FlatMapInPlaceVec`.
impl<V: FlatMapInPlaceVec> FlatMapInPlace<V::Elem> for V {
    fn flat_map_in_place<F, I>(&mut self, mut f: F)
    where
        F: FnMut(V::Elem) -> I,
        I: IntoIterator<Item = V::Elem>,
    {
        struct LeakGuard<'a, V: FlatMapInPlaceVec>(&'a mut V);

        impl<'a, V: FlatMapInPlaceVec> Drop for LeakGuard<'a, V> {
            fn drop(&mut self) {
                unsafe {
                    // Leak all elements in case of panic.
                    self.0.set_len(0);
                }
            }
        }

        let guard = LeakGuard(self);

        let mut read_i = 0;
        let mut write_i = 0;
        unsafe {
            while read_i < guard.0.len() {
                // Move the read_i'th item out of the vector and map it to an iterator.
                let e = ptr::read(guard.0.as_ptr().add(read_i));
                let iter = f(e).into_iter();
                read_i += 1;

                for e in iter {
                    if write_i < read_i {
                        ptr::write(guard.0.as_mut_ptr().add(write_i), e);
                        write_i += 1;
                    } else {
                        // If this is reached we ran out of space in the middle of the vector.
                        // However, the vector is in a valid state here, so we just do a somewhat
                        // inefficient insert.
                        guard.0.insert(write_i, e);

                        read_i += 1;
                        write_i += 1;
                    }
                }
            }

            // `write_i` tracks the number of actually written new items.
            guard.0.set_len(write_i);

            // `vec` is in a sane state again. Prevent the LeakGuard from leaking the data.
            mem::forget(guard);
        }
    }
}

// A vec-like type must implement these operations to support `flat_map_in_place`.
pub trait FlatMapInPlaceVec {
    type Elem;

    fn len(&self) -> usize;
    unsafe fn set_len(&mut self, len: usize);
    fn as_ptr(&self) -> *const Self::Elem;
    fn as_mut_ptr(&mut self) -> *mut Self::Elem;
    fn insert(&mut self, idx: usize, elem: Self::Elem);
}

impl<T> FlatMapInPlaceVec for Vec<T> {
    type Elem = T;

    fn len(&self) -> usize {
        self.len()
    }

    unsafe fn set_len(&mut self, len: usize) {
        unsafe {
            self.set_len(len);
        }
    }

    fn as_ptr(&self) -> *const Self::Elem {
        self.as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut Self::Elem {
        self.as_mut_ptr()
    }

    fn insert(&mut self, idx: usize, elem: Self::Elem) {
        self.insert(idx, elem);
    }
}

impl<T> FlatMapInPlaceVec for ThinVec<T> {
    type Elem = T;

    fn len(&self) -> usize {
        self.len()
    }

    unsafe fn set_len(&mut self, len: usize) {
        unsafe {
            self.set_len(len);
        }
    }

    fn as_ptr(&self) -> *const Self::Elem {
        self.as_slice().as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut Self::Elem {
        self.as_mut_slice().as_mut_ptr()
    }

    fn insert(&mut self, idx: usize, elem: Self::Elem) {
        self.insert(idx, elem);
    }
}

impl<T, const N: usize> FlatMapInPlaceVec for SmallVec<[T; N]> {
    type Elem = T;

    fn len(&self) -> usize {
        self.len()
    }

    unsafe fn set_len(&mut self, len: usize) {
        unsafe {
            self.set_len(len);
        }
    }

    fn as_ptr(&self) -> *const Self::Elem {
        self.as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut Self::Elem {
        self.as_mut_ptr()
    }

    fn insert(&mut self, idx: usize, elem: Self::Elem) {
        self.insert(idx, elem);
    }
}
