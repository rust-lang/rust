// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME: talk about offset, copy_memory, copy_nonoverlapping_memory

//! Operations on unsafe pointers, `*const T`, and `*mut T`.
//!
//! Working with unsafe pointers in Rust is uncommon,
//! typically limited to a few patterns.
//!
//! Use the [`null` function](fn.null.html) to create null pointers,
//! the [`is_null`](trait.RawPtr.html#tymethod.is_null)
//! and [`is_not_null`](trait.RawPtr.html#method.is_not_null)
//! methods of the [`RawPtr` trait](trait.RawPtr.html) to check for null.
//! The `RawPtr` trait is imported by the prelude, so `is_null` etc.
//! work everywhere. The `RawPtr` also defines the `offset` method,
//! for pointer math.
//!
//! # Common ways to create unsafe pointers
//!
//! ## 1. Coerce a reference (`&T`) or mutable reference (`&mut T`).
//!
//! ```
//! let my_num: int = 10;
//! let my_num_ptr: *const int = &my_num;
//! let mut my_speed: int = 88;
//! let my_speed_ptr: *mut int = &mut my_speed;
//! ```
//!
//! This does not take ownership of the original allocation
//! and requires no resource management later,
//! but you must not use the pointer after its lifetime.
//!
//! ## 2. Transmute an owned box (`Box<T>`).
//!
//! The `transmute` function takes, by value, whatever it's given
//! and returns it as whatever type is requested, as long as the
//! types are the same size. Because `Box<T>` and `*mut T` have the same
//! representation they can be trivially,
//! though unsafely, transformed from one type to the other.
//!
//! ```
//! use std::mem;
//!
//! unsafe {
//!     let my_num: Box<int> = box 10;
//!     let my_num: *const int = mem::transmute(my_num);
//!     let my_speed: Box<int> = box 88;
//!     let my_speed: *mut int = mem::transmute(my_speed);
//!
//!     // By taking ownership of the original `Box<T>` though
//!     // we are obligated to transmute it back later to be destroyed.
//!     drop(mem::transmute::<_, Box<int>>(my_speed));
//!     drop(mem::transmute::<_, Box<int>>(my_num));
//! }
//! ```
//!
//! Note that here the call to `drop` is for clarity - it indicates
//! that we are done with the given value and it should be destroyed.
//!
//! ## 3. Get it from C.
//!
//! ```
//! extern crate libc;
//!
//! use std::mem;
//!
//! fn main() {
//!     unsafe {
//!         let my_num: *mut int = libc::malloc(mem::size_of::<int>() as libc::size_t) as *mut int;
//!         if my_num.is_null() {
//!             fail!("failed to allocate memory");
//!         }
//!         libc::free(my_num as *mut libc::c_void);
//!     }
//! }
//! ```
//!
//! Usually you wouldn't literally use `malloc` and `free` from Rust,
//! but C APIs hand out a lot of pointers generally, so are a common source
//! of unsafe pointers in Rust.

use mem;
use clone::Clone;
use intrinsics;
use iter::range;
use option::{Some, None, Option};

use cmp::{PartialEq, Eq, PartialOrd, Equiv, Ordering, Less, Equal, Greater};

pub use intrinsics::copy_memory;
pub use intrinsics::copy_nonoverlapping_memory;
pub use intrinsics::set_memory;

/// Create a null pointer.
///
/// # Example
///
/// ```
/// use std::ptr;
///
/// let p: *const int = ptr::null();
/// assert!(p.is_null());
/// ```
#[inline]
#[unstable = "may need a different name after pending changes to pointer types"]
pub fn null<T>() -> *const T { 0 as *const T }

/// Create an unsafe mutable null pointer.
///
/// # Example
///
/// ```
/// use std::ptr;
///
/// let p: *mut int = ptr::mut_null();
/// assert!(p.is_null());
/// ```
#[inline]
#[unstable = "may need a different name after pending changes to pointer types"]
pub fn mut_null<T>() -> *mut T { 0 as *mut T }

/// Zeroes out `count * size_of::<T>` bytes of memory at `dst`
#[inline]
#[experimental = "uncertain about naming and semantics"]
#[allow(experimental)]
pub unsafe fn zero_memory<T>(dst: *mut T, count: uint) {
    set_memory(dst, 0, count);
}

/// Swap the values at two mutable locations of the same type, without
/// deinitialising either. They may overlap.
#[inline]
#[unstable]
pub unsafe fn swap<T>(x: *mut T, y: *mut T) {
    // Give ourselves some scratch space to work with
    let mut tmp: T = mem::uninitialized();
    let t: *mut T = &mut tmp;

    // Perform the swap
    copy_nonoverlapping_memory(t, &*x, 1);
    copy_memory(x, &*y, 1); // `x` and `y` may overlap
    copy_nonoverlapping_memory(y, &*t, 1);

    // y and t now point to the same thing, but we need to completely forget `tmp`
    // because it's no longer relevant.
    mem::forget(tmp);
}

/// Replace the value at a mutable location with a new one, returning the old
/// value, without deinitialising either.
#[inline]
#[unstable]
pub unsafe fn replace<T>(dest: *mut T, mut src: T) -> T {
    mem::swap(mem::transmute(dest), &mut src); // cannot overlap
    src
}

/// Reads the value from `*src` and returns it.
#[inline(always)]
#[unstable]
pub unsafe fn read<T>(src: *const T) -> T {
    let mut tmp: T = mem::uninitialized();
    copy_nonoverlapping_memory(&mut tmp, src, 1);
    tmp
}

/// Reads the value from `*src` and nulls it out.
/// This currently prevents destructors from executing.
#[inline(always)]
#[experimental]
#[allow(experimental)]
pub unsafe fn read_and_zero<T>(dest: *mut T) -> T {
    // Copy the data out from `dest`:
    let tmp = read(&*dest);

    // Now zero out `dest`:
    zero_memory(dest, 1);

    tmp
}

/// Unsafely overwrite a memory location with the given value without destroying
/// the old value.
///
/// This operation is unsafe because it does not destroy the previous value
/// contained at the location `dst`. This could leak allocations or resources,
/// so care must be taken to previously deallocate the value at `dst`.
#[inline]
#[unstable]
pub unsafe fn write<T>(dst: *mut T, src: T) {
    intrinsics::move_val_init(&mut *dst, src)
}

/// Given a *const *const T (pointer to an array of pointers),
/// iterate through each *const T, up to the provided `len`,
/// passing to the provided callback function
#[deprecated = "old-style iteration. use a loop and RawPtr::offset"]
pub unsafe fn array_each_with_len<T>(arr: *const *const T, len: uint,
                                     cb: |*const T|) {
    if arr.is_null() {
        fail!("ptr::array_each_with_len failure: arr input is null pointer");
    }
    //let start_ptr = *arr;
    for e in range(0, len) {
        let n = arr.offset(e as int);
        cb(*n);
    }
}

/// Given a null-pointer-terminated *const *const T (pointer to
/// an array of pointers), iterate through each *const T,
/// passing to the provided callback function
///
/// # Safety Note
///
/// This will only work with a null-terminated
/// pointer array.
#[deprecated = "old-style iteration. use a loop and RawPtr::offset"]
#[allow(deprecated)]
pub unsafe fn array_each<T>(arr: *const  *const T, cb: |*const T|) {
    if arr.is_null()  {
        fail!("ptr::array_each_with_len failure: arr input is null pointer");
    }
    let len = buf_len(arr);
    array_each_with_len(arr, len, cb);
}

/// Return the offset of the first null pointer in `buf`.
#[inline]
#[deprecated = "use a loop and RawPtr::offset"]
#[allow(deprecated)]
pub unsafe fn buf_len<T>(buf: *const *const T) -> uint {
    position(buf, |i| *i == null())
}

/// Return the first offset `i` such that `f(buf[i]) == true`.
#[inline]
#[deprecated = "old-style iteration. use a loop and RawPtr::offset"]
pub unsafe fn position<T>(buf: *const T, f: |&T| -> bool) -> uint {
    let mut i = 0;
    loop {
        if f(&(*buf.offset(i as int))) { return i; }
        else { i += 1; }
    }
}

/// Methods on raw pointers
pub trait RawPtr<T> {
    /// Returns the null pointer.
    fn null() -> Self;

    /// Returns true if the pointer is equal to the null pointer.
    fn is_null(&self) -> bool;

    /// Returns true if the pointer is not equal to the null pointer.
    fn is_not_null(&self) -> bool { !self.is_null() }

    /// Returns the value of this pointer (ie, the address it points to)
    fn to_uint(&self) -> uint;

    /// Returns `None` if the pointer is null, or else returns a reference to the
    /// value wrapped in `Some`.
    ///
    /// # Safety Notes
    ///
    /// While this method and its mutable counterpart are useful for null-safety,
    /// it is important to note that this is still an unsafe operation because
    /// the returned value could be pointing to invalid memory.
    unsafe fn as_ref<'a>(&self) -> Option<&'a T>;

    /// A synonym for `as_ref`, except with incorrect lifetime semantics
    #[deprecated="Use `as_ref` instead"]
    unsafe fn to_option<'a>(&'a self) -> Option<&'a T> {
        mem::transmute(self.as_ref())
    }

    /// Calculates the offset from a pointer. The offset *must* be in-bounds of
    /// the object, or one-byte-past-the-end.  `count` is in units of T; e.g. a
    /// `count` of 3 represents a pointer offset of `3 * sizeof::<T>()` bytes.
    unsafe fn offset(self, count: int) -> Self;
}

/// Methods on mutable raw pointers
pub trait RawMutPtr<T>{
    /// Returns `None` if the pointer is null, or else returns a mutable reference
    /// to the value wrapped in `Some`. As with `as_ref`, this is unsafe because
    /// it cannot verify the validity of the returned pointer.
    unsafe fn as_mut<'a>(&self) -> Option<&'a mut T>;
}

impl<T> RawPtr<T> for *const T {
    #[inline]
    fn null() -> *const T { null() }

    #[inline]
    fn is_null(&self) -> bool { *self == RawPtr::null() }

    #[inline]
    fn to_uint(&self) -> uint { *self as uint }

    #[inline]
    unsafe fn offset(self, count: int) -> *const T {
        intrinsics::offset(self, count)
    }

    #[inline]
    unsafe fn as_ref<'a>(&self) -> Option<&'a T> {
        if self.is_null() {
            None
        } else {
            Some(&**self)
        }
    }
}

impl<T> RawPtr<T> for *mut T {
    #[inline]
    fn null() -> *mut T { mut_null() }

    #[inline]
    fn is_null(&self) -> bool { *self == RawPtr::null() }

    #[inline]
    fn to_uint(&self) -> uint { *self as uint }

    #[inline]
    unsafe fn offset(self, count: int) -> *mut T {
        intrinsics::offset(self as *const T, count) as *mut T
    }

    #[inline]
    unsafe fn as_ref<'a>(&self) -> Option<&'a T> {
        if self.is_null() {
            None
        } else {
            Some(&**self)
        }
    }
}

impl<T> RawMutPtr<T> for *mut T {
    #[inline]
    unsafe fn as_mut<'a>(&self) -> Option<&'a mut T> {
        if self.is_null() {
            None
        } else {
            Some(&mut **self)
        }
    }
}

// Equality for pointers
impl<T> PartialEq for *const T {
    #[inline]
    fn eq(&self, other: &*const T) -> bool {
        *self == *other
    }
    #[inline]
    fn ne(&self, other: &*const T) -> bool { !self.eq(other) }
}

impl<T> Eq for *const T {}

impl<T> PartialEq for *mut T {
    #[inline]
    fn eq(&self, other: &*mut T) -> bool {
        *self == *other
    }
    #[inline]
    fn ne(&self, other: &*mut T) -> bool { !self.eq(other) }
}

impl<T> Eq for *mut T {}

// Equivalence for pointers
impl<T> Equiv<*mut T> for *const T {
    fn equiv(&self, other: &*mut T) -> bool {
        self.to_uint() == other.to_uint()
    }
}

impl<T> Equiv<*const T> for *mut T {
    fn equiv(&self, other: &*const T) -> bool {
        self.to_uint() == other.to_uint()
    }
}

impl<T> Clone for *const T {
    #[inline]
    fn clone(&self) -> *const T {
        *self
    }
}

impl<T> Clone for *mut T {
    #[inline]
    fn clone(&self) -> *mut T {
        *self
    }
}

// Equality for extern "C" fn pointers
mod externfnpointers {
    use mem;
    use cmp::PartialEq;

    impl<_R> PartialEq for extern "C" fn() -> _R {
        #[inline]
        fn eq(&self, other: &extern "C" fn() -> _R) -> bool {
            let self_: *const () = unsafe { mem::transmute(*self) };
            let other_: *const () = unsafe { mem::transmute(*other) };
            self_ == other_
        }
    }
    macro_rules! fnptreq(
        ($($p:ident),*) => {
            impl<_R,$($p),*> PartialEq for extern "C" fn($($p),*) -> _R {
                #[inline]
                fn eq(&self, other: &extern "C" fn($($p),*) -> _R) -> bool {
                    let self_: *const () = unsafe { mem::transmute(*self) };

                    let other_: *const () = unsafe { mem::transmute(*other) };
                    self_ == other_
                }
            }
        }
    )
    fnptreq!(A)
    fnptreq!(A,B)
    fnptreq!(A,B,C)
    fnptreq!(A,B,C,D)
    fnptreq!(A,B,C,D,E)
}

// Comparison for pointers
impl<T> PartialOrd for *const T {
    #[inline]
    fn partial_cmp(&self, other: &*const T) -> Option<Ordering> {
        if self < other {
            Some(Less)
        } else if self == other {
            Some(Equal)
        } else {
            Some(Greater)
        }
    }

    #[inline]
    fn lt(&self, other: &*const T) -> bool { *self < *other }

    #[inline]
    fn le(&self, other: &*const T) -> bool { *self <= *other }

    #[inline]
    fn gt(&self, other: &*const T) -> bool { *self > *other }

    #[inline]
    fn ge(&self, other: &*const T) -> bool { *self >= *other }
}

impl<T> PartialOrd for *mut T {
    #[inline]
    fn partial_cmp(&self, other: &*mut T) -> Option<Ordering> {
        if self < other {
            Some(Less)
        } else if self == other {
            Some(Equal)
        } else {
            Some(Greater)
        }
    }

    #[inline]
    fn lt(&self, other: &*mut T) -> bool { *self < *other }

    #[inline]
    fn le(&self, other: &*mut T) -> bool { *self <= *other }

    #[inline]
    fn gt(&self, other: &*mut T) -> bool { *self > *other }

    #[inline]
    fn ge(&self, other: &*mut T) -> bool { *self >= *other }
}
