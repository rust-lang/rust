use smallvec::{Array, SmallVec};
use std::ptr;
use thin_vec::ThinVec;

pub trait MapInPlace<T>: Sized {
    fn map_in_place<F>(&mut self, mut f: F)
    where
        F: FnMut(T) -> T,
    {
        self.flat_map_in_place(|e| Some(f(e)))
    }

    fn flat_map_in_place<F, I>(&mut self, f: F)
    where
        F: FnMut(T) -> I,
        I: IntoIterator<Item = T>;
}

// The implementation of this method is syntactically identical for all the
// different vector types.
macro_rules! flat_map_in_place {
    () => {
        fn flat_map_in_place<F, I>(&mut self, mut f: F)
        where
            F: FnMut(T) -> I,
            I: IntoIterator<Item = T>,
        {
            let mut read_i = 0;
            let mut write_i = 0;
            unsafe {
                let mut old_len = self.len();
                self.set_len(0); // make sure we just leak elements in case of panic

                while read_i < old_len {
                    // move the read_i'th item out of the vector and map it
                    // to an iterator
                    let e = ptr::read(self.as_ptr().add(read_i));
                    let iter = f(e).into_iter();
                    read_i += 1;

                    for e in iter {
                        if write_i < read_i {
                            ptr::write(self.as_mut_ptr().add(write_i), e);
                            write_i += 1;
                        } else {
                            // If this is reached we ran out of space
                            // in the middle of the vector.
                            // However, the vector is in a valid state here,
                            // so we just do a somewhat inefficient insert.
                            self.set_len(old_len);
                            self.insert(write_i, e);

                            old_len = self.len();
                            self.set_len(0);

                            read_i += 1;
                            write_i += 1;
                        }
                    }
                }

                // write_i tracks the number of actually written new items.
                self.set_len(write_i);
            }
        }
    };
}

impl<T> MapInPlace<T> for Vec<T> {
    flat_map_in_place!();
}

impl<T, A: Array<Item = T>> MapInPlace<T> for SmallVec<A> {
    flat_map_in_place!();
}

impl<T> MapInPlace<T> for ThinVec<T> {
    flat_map_in_place!();
}
