use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ptr;
use std::ptr::NonNull;

use crate::vec_extractor::raw_slice::RawSliceIter;

mod raw_slice;

/// A cursor-like API for vec; generalization of `Vec::drain` and `Vec::extract_if`.
pub struct Extractor<'e, T: 'e> {
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
    kept_end: NonNull<T>,
    rest: RawSliceIter<'e, T>,
    vec: NonNull<Vec<T>>,
    _ghost: PhantomData<&'e mut Vec<T>>,
}

impl<T> Drop for Extractor<'_, T> {
    fn drop(&mut self) {
        unsafe {
            let vec = self.vec.as_mut();

            let kept = self.kept_end.as_ptr().offset_from_unsigned(vec.as_ptr());
            let rest = self.rest.len();
            let len = kept + rest;

            ptr::copy(self.rest.as_slice().as_ptr().cast(), self.kept_end.as_ptr(), rest);

            vec.set_len(len);
        }
    }
}

pub struct Entry<'e, 'a, T> {
    // Invariants:
    // - `val` points into `*extractor.vec`, is valid for reads & writes,
    //   does not overlap with `extractor.rest`, and is `>= extractor.kept_end`
    extractor: &'a mut Extractor<'e, T>,
    val: &'a mut T,
}

impl<T> Drop for Entry<'_, '_, T> {
    fn drop(&mut self) {
        unsafe {
            let kept_end = self.extractor.kept_end;
            ptr::copy(ptr::from_ref(self.val), kept_end.as_ptr(), 1);
            self.extractor.kept_end = kept_end.add(1);
        }
    }
}

pub struct Hole<'e, 'a, T> {
    // Invariants:
    // - `extractor.kept_end` does not overlap with `extractor.rest`
    extractor: &'a mut Extractor<'e, T>,
}

impl<'e, T> Extractor<'e, T> {
    pub fn new(v: &'e mut Vec<T>) -> Self {
        let len = v.len();

        unsafe {
            // Safety: setting length to 0 is always sound
            v.set_len(0);

            // Safety: this is equivalent to `v.as_slice()`, but before setting
            //         the length to 0
            // TODO
            let slice = NonNull::slice_from_raw_parts(NonNull::new(v.as_mut_ptr()).unwrap(), len);

            // Safety:
            // - `vec` points to a vector with length 0 and is valid
            // - `rest` and `kept_end` point into `*vec`'s allocation
            // - the range between `vec.as_ref().as_ptr()` and `kept_end` is
            //   empty (and this initialized)
            // - `kept_end <= rest.as_ptr()` (they are equal)
            Self {
                kept_end: slice.cast(),
                rest: RawSliceIter::new(slice),
                vec: NonNull::from(v),
                _ghost: PhantomData,
            }
        }
    }

    pub fn entry(&mut self) -> Option<Entry<'e, '_, T>> {
        let val = self.rest.next()?;

        // Safety: `val` points into extractor's vec, since `rest` does.
        //         it is valid for reads & writes (as we got it from the vec),
        //         and does not overlap with `rest` (we just got it from there)
        Some(Entry { extractor: self, val: unsafe { { val }.as_mut() } })
    }

    pub fn drop_rest(mut self) {
        unsafe {
            std::ptr::drop_in_place(self.rest.as_slice().as_ptr());
            self.rest = <_>::default();
        }
    }
}

impl<'e, 'a, T> Entry<'e, 'a, T> {
    pub fn take(self) -> (T, Hole<'e, 'a, T>) {
        unsafe {
            let (extractor, val) = self.into_raw_parts();
            let val = NonNull::from(val).read();
            (val, Hole { extractor })
        }
    }

    pub fn keep(self) { /* drop is equivalent to keeping */
    }

    fn into_raw_parts(self) -> (&'a mut Extractor<'e, T>, &'a mut T) {
        let this = ManuallyDrop::new(self);

        // Safety: moving out a field
        unsafe { (ptr::from_ref(&this.extractor).read(), ptr::from_ref(&this.val).read()) }
    }
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

#[test]
fn smoke() {
    let mut v = (0..100).into_iter().map(|x| (x, x.to_string())).collect::<Vec<_>>();

    let mut e = Extractor::new(&mut v);

    while let Some(entry) = e.entry() {
        let (elem, hole) = entry.take();

        if elem.0 > 50 {
            break;
        }

        if elem.0 % 3 == 0 {
            hole.fill((1000, "meow".to_owned()));
        } else if elem.0 % 3 == 1 {
            hole.fill(elem);
        }
    }

    e.drop_rest();
    dbg!(v);
}
