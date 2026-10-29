#![expect(unreachable_pub)] // to be moved to a better place (like std?)
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ptr;
use std::ptr::NonNull;

use crate::vec_extractor::raw_slice::RawSliceIter;

mod raw_slice;

/// A cursor-like API for vec; generalization of `Vec::drain` and `Vec::extract_if`.
pub struct Extractor<'e, T> {
    // Invariants:
    // - `vec` points to a vector with length set to 0, and is valid for reads &
    //   writes
    // - `rest` and `kept_end` point into `*vec`'s allocation
    // - the range between `vec.as_ref().as_ptr()` and `kept_end` is intialized
    // - `kept_end <= rest.as_ptr()`
    //
    // Essentially                    kept elements              rest
    //                       ________/                __________/
    //                      /        \               /          \
    //     vec allocation: [                                                   ]
    //                     ^         ^\_____________/^          ^\____________/
    // Extractor {         |         |       |       |          |       |
    //     vec: --> Vec {  |         |    extracted  |          |   spare vec
    //                ptr -+         |    elements   |          |   capacity
    //                cap: N,        |               |          |
    //                len: 0,        |               |          |
    //              }                |               |          |
    //     kept_end: ----------------+               |          |
    //                                               |          |
    //     rest: ------------------------------------+----------+
    //
    // }
    //
    /// Points past-the-end of the elements that were kept in the vector.
    kept_end: NonNull<T>,

    /// Elements that haven't been accessed yet.
    rest: RawSliceIter<'e, T>,

    /// The original vector (length set to 0 in order to avoid UB when `Extractor` is leaked).
    vec: NonNull<Vec<T>>,

    /// We essentially hold a `&mut Vec<T>`. The only reason `vec` is not literally of that type
    /// is to prevent accidentally dereferencing the vector, which would invalidate `self.rest`.
    _ghost: PhantomData<&'e mut Vec<T>>,
}

impl<'e, T> Extractor<'e, T> {
    pub fn new(v: &'e mut Vec<T>) -> Self {
        let len = v.len();

        unsafe {
            // Safety: setting length to 0 is always sound
            v.set_len(0);

            // Safety: this is equivalent to `v.as_slice()`, but before setting the length to 0
            let slice = NonNull::slice_from_raw_parts(NonNull::new_unchecked(v.as_mut_ptr()), len);

            // NB: it is important that both of these are derived from the same pointer.
            //     later we are gonna turn `rest.next()` into a mutable reference, and we need that
            //     to not invalidate `kept_end`.
            let rest = RawSliceIter::new(slice);
            let kept_end = slice.cast();

            // Safety:
            // - `vec` points to a vector with length 0 and is valid
            // - `rest` and `kept_end` point into `*vec`'s allocation
            // - the range between `vec.as_ref().as_ptr()` and `kept_end` is empty
            //   (and thus initialized)
            // - `kept_end <= rest.as_ptr()` (they are equal)
            Self {
                kept_end: slice.cast(),
                rest: RawSliceIter::new(slice),
                vec: NonNull::from(v),
                _ghost: PhantomData,
            }
        }
    }

    /// Returns an entry for the "current" element.
    pub fn entry(&mut self) -> Option<Entry<'e, '_, T>> {
        let mut val = self.rest.next()?;

        // Safety: `val` points into extractor's vec, since `rest` does.
        //         it is valid for reads & writes (as we got it from the vec),
        //         and does not overlap with `rest` (we just got it from there)
        Some(Entry { extractor: self, val: unsafe { val.as_mut() } })
    }

    /// Drops the rest of the elements, that haven't been inspected yet.
    pub fn drop_rest(&mut self) {
        unsafe {
            // Safety: `self.rest` points to initialized values
            std::ptr::drop_in_place(self.rest.as_slice().as_ptr());

            // NB: remove the elements we just dropped
            self.rest = RawSliceIter::default();
        }
    }
}

impl<T> Drop for Extractor<'_, T> {
    /// Keeps the elements that haven't been seen in the original vector.
    ///
    /// Use [`Extractor::drop_rest`] to avoid this.
    fn drop(&mut self) {
        // Safety:
        // we do not dereference the vec to a slice, being careful to only use `as_ptr`
        // and `set_len`.
        let vec = unsafe { self.vec.as_mut() };

        // Safety: `kept_end` points into `vec`'s allocation
        let kept = unsafe { self.kept_end.as_ptr().offset_from_unsigned(vec.as_ptr()) };
        let rest = self.rest.len();
        let len = kept + rest;

        // Safety:
        // - `kept_end` is valid for writes, per the type's invariants
        // - `rest.as_slice()` is valid for reads for `rest.len()` elements, per the type's invariants
        ptr::copy(self.rest.as_slice().as_ptr().cast(), self.kept_end.as_ptr(), rest);

        // Safety:
        // After the manipulations above, `len` elements at the start of the vec *are* intialized.
        vec.set_len(len);
    }
}

pub struct Entry<'e, 'a, T> {
    // Invariants:
    // - `val` points into `*extractor.vec`, does not overlap with `extractor.rest`,
    //   and is `>= extractor.kept_end`
    // - The entry essentially owns the value
    //   - If the entry is dropped, the value is returned to the vec
    //   - If the entry is leaked, the value is leaked as well
    extractor: &'a mut Extractor<'e, T>,
    val: &'a mut T,
}

impl<T> Drop for Entry<'_, '_, T> {
    fn drop(&mut self) {
        unsafe {
            let kept_end = self.extractor.kept_end;

            // Safety
            //
            // TODO: this is unsound apparently lmao
            //
            // if entry is passed through a function, that adds a protector on `val`, which this copy invalidates
            //
            // - `self.val` is a reference and thus valid for reads
            // - `kept_end` is valid for writes, per `extractor`'s invariants
            ptr::copy(self.val, kept_end.as_ptr(), 1);

            // Safety:
            // - `kept_end <= val`, thus `kept_end+1` is at most one-past-the-end of the allocation
            // - new `kept_end` value is still `<= rest.as_ptr()`, since `val` and `rest` don't overlap
            self.extractor.kept_end = kept_end.add(1);
        }
    }
}

impl<'e, 'a, T> Entry<'e, 'a, T> {
    /// Takes the element out of the vec, and returns a "hole" to where the element can be returned.
    pub fn take(self) -> (T, Hole<'e, 'a, T>) {
        unsafe {
            let (extractor, val) = self.into_raw_parts();
            let val = NonNull::from(val).read();
            (val, Hole { extractor })
        }
    }

    /// Takes the element out of the vec.
    ///
    /// Equivalent to [`Entry::take`], but doesn't return the hole.
    pub fn into_value(self) -> T {
        self.take().0
    }

    /// Keep the element in the vec (equivalent to dropping the entry).
    pub fn keep(self) {}

    pub fn as_ref(&self) -> &T {
        &self.val
    }

    pub fn as_mut(&mut self) -> &mut T {
        self.val
    }

    fn into_raw_parts(self) -> (&'a mut Extractor<'e, T>, &'a mut T) {
        let this = ManuallyDrop::new(self);

        // Safety: moving out a field
        unsafe { (ptr::from_ref(&this.extractor).read(), ptr::from_ref(&this.val).read()) }
    }
}

pub struct Hole<'e, 'a, T> {
    // Invariants:
    // - `extractor.kept_end` does not overlap with `extractor.rest`
    //   (in other words, there is a space where an element can be put, at `kept_end`)
    extractor: &'a mut Extractor<'e, T>,
}

impl<'e, 'a, T> Hole<'e, 'a, T> {
    pub fn fill(self, val: T) -> Entry<'e, 'a, T> {
        unsafe {
            let end = self.extractor.kept_end;
            end.write(val);

            Entry { extractor: self.extractor, val: { end }.as_mut() }
        }
    }
}

#[cfg(test)]
mod tests;
