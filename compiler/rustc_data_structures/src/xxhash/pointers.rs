//! This module defines Ptr and PtrMut, which are pointers into slices that get their
//! bounds checked only when debug assertions are enabled.

// yuck.
macro_rules! slice_to_ptr {
    ($slice:expr, Ptr) => {
        $slice.as_ptr()
    };
    ($slice:expr, PtrMut) => {
        $slice.as_mut_ptr()
    };
}

macro_rules! define_ptr {
    ($name:ident, $ptr_ty:ty, $from_slice_ty:ty) => {
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        pub struct $name<T: Copy> {
            ptr: $ptr_ty,
            #[cfg(debug_assertions)]
            lower_bound: usize,
            #[cfg(debug_assertions)]
            upper_bound: usize,
        }

        impl<T: Copy> From<$from_slice_ty> for $name<T> {
            #[inline]
            fn from(slice: $from_slice_ty) -> Self {
                #[cfg(debug_assertions)]
                let bounds = slice.as_ptr_range();

                Self {
                    ptr: slice_to_ptr!(slice, $name),
                    #[cfg(debug_assertions)]
                    lower_bound: bounds.start as usize,
                    #[cfg(debug_assertions)]
                    upper_bound: bounds.end as usize,
                }
            }
        }

        impl<T: Copy> $name<T> {
            #[inline]
            #[allow(unused)]
            pub fn cast<U: Copy>(self) -> $name<U> {
                debug_assert!({
                    use std::cmp::{max, min};
                    use std::mem::size_of;
                    let t_size = size_of::<T>();
                    let u_size = size_of::<U>();
                    max(t_size, u_size) % min(t_size, u_size) == 0
                });

                $name {
                    ptr: self.ptr.cast(),
                    #[cfg(debug_assertions)]
                    lower_bound: self.lower_bound,
                    #[cfg(debug_assertions)]
                    upper_bound: self.upper_bound,
                }
            }

            #[inline]
            pub fn raw(self) -> $ptr_ty {
                self.ptr
            }

            #[inline]
            fn addr(&self) -> usize {
                self.ptr as usize
            }

            #[allow(unused)]
            #[inline]
            pub fn read(&self) -> T {
                #[cfg(debug_assertions)]
                {
                    assert!(self.addr() >= self.lower_bound);
                    assert!(self.addr() < self.upper_bound);
                    assert!(self.addr() % std::mem::align_of::<T>() == 0);
                }

                unsafe { *self.ptr }
            }

            #[inline]
            pub fn offset(self, offset: usize) -> Self {
                #[cfg(debug_assertions)]
                {
                    assert!(self.addr() + offset <= self.upper_bound);
                }

                unsafe {
                    Self {
                        ptr: self.ptr.add(offset),
                        #[cfg(debug_assertions)]
                        lower_bound: self.lower_bound,
                        #[cfg(debug_assertions)]
                        upper_bound: self.upper_bound,
                    }
                }
            }

            #[allow(unused)]
            #[inline]
            pub fn negative_offset(self, offset: usize) -> Self {
                #[cfg(debug_assertions)]
                {
                    assert!(self.addr() - offset >= self.lower_bound);
                }
                unsafe {
                    Self {
                        ptr: self.ptr.sub(offset),
                        #[cfg(debug_assertions)]
                        lower_bound: self.lower_bound,
                        #[cfg(debug_assertions)]
                        upper_bound: self.upper_bound,
                    }
                }
            }

            #[allow(unused)]
            #[inline]
            pub fn distance(self, other: Self) -> usize {
                debug_assert!(self.addr() >= other.addr());
                self.addr() - other.addr()
            }

            #[cfg(debug_assertions)]
            pub fn assert_contains_next_n_bytes(self, byte_count: usize) {
                assert!(self.addr() >= self.lower_bound);
                assert!(self.addr() + byte_count <= self.upper_bound);
            }
        }
    };
}

define_ptr!(Ptr, *const T, &[T]);
define_ptr!(PtrMut, *mut T, &mut [T]);

impl<T: Copy> PtrMut<T> {
    #[inline]
    pub fn to_const_ptr(self) -> Ptr<T> {
        return Ptr {
            ptr: self.ptr as *const T,
            #[cfg(debug_assertions)]
            lower_bound: self.lower_bound,
            #[cfg(debug_assertions)]
            upper_bound: self.upper_bound,
        };
    }

    #[allow(unused)]
    #[inline]
    pub fn write(self, value: T) {
        #[cfg(debug_assertions)]
        {
            assert!(self.addr() >= self.lower_bound);
            assert!(self.addr() < self.upper_bound);
            assert_eq!(self.addr() % std::mem::align_of::<T>(), 0);
        }

        unsafe {
            *self.ptr = value;
        }
    }
}
