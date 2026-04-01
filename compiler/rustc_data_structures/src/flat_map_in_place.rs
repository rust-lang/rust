use std::{mem, ptr};

use smallvec::{Array, SmallVec};
use thin_vec::ThinVec;

pub trait FlatMapInPlace<T>: Sized {
    fn flat_map_in_place<F, I>(&mut self, f: F)
    where
        F: FnMut(T) -> I,
        I: IntoIterator<Item = T>;
}

// The implementation of this method is syntactically identical for all the
// different vector types.
macro_rules! flat_map_in_place {
    ($vec:ident $( where T: $bound:path)?) => {
        fn flat_map_in_place<F, I>(&mut self, mut f: F)
        where
            F: FnMut(T) -> I,
            I: IntoIterator<Item = T>,
        {
            struct LeakGuard<'a, T $(: $bound)?>(&'a mut $vec<T>);

            impl<'a, T $(: $bound)?> Drop for LeakGuard<'a, T> {
                fn drop(&mut self) {
                    unsafe {
                        self.0.set_len(0); // make sure we just leak elements in case of panic
                    }
                }
            }

            let this = LeakGuard(self);

            let mut read_i = 0;
            let mut write_i = 0;
            unsafe {
                while read_i < this.0.len() {
                    // move the read_i'th item out of the vector and map it
                    // to an iterator
                    let e = ptr::read(this.0.as_ptr().add(read_i));
                    let iter = f(e).into_iter();
                    read_i += 1;

                    for e in iter {
                        if write_i < read_i {
                            ptr::write(this.0.as_mut_ptr().add(write_i), e);
                            write_i += 1;
                        } else {
                            // If this is reached we ran out of space
                            // in the middle of the vector.
                            // However, the vector is in a valid state here,
                            // so we just do a somewhat inefficient insert.
                            this.0.insert(write_i, e);

                            read_i += 1;
                            write_i += 1;
                        }
                    }
                }

                // write_i tracks the number of actually written new items.
                this.0.set_len(write_i);

                // The ThinVec is in a sane state again. Prevent the LeakGuard from leaking the data.
                mem::forget(this);
            }
        }
    };
}

impl<T> FlatMapInPlace<T> for Vec<T> {
    flat_map_in_place!(Vec);
}

impl<T, A: Array<Item = T>> FlatMapInPlace<T> for SmallVec<A> {
    flat_map_in_place!(SmallVec where T: Array);
}

impl<T> FlatMapInPlace<T> for ThinVec<T> {
    flat_map_in_place!(ThinVec);
}
