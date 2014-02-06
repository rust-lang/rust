// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * Volatile Module.
 *
 * The order of variables, and their execution isn't guaranteed. The compiler may optimize
 * statements to make things more efficient. However, when you're working with memory that
 * other devices/programs may also be using, you need to be specific about how and when
 * that memory is read and written.
 *
 * That's where Volatile memory comes into play. They tell the compiler not to move the
 * order in which the memory is read or written.
 *
 *
 * http://en.wikipedia.org/wiki/Volatile_variable
 * http://llvm.org/docs/LangRef.html#volatile-memory-accesses
 *
 *
 * This module provides a number of different types and implementations to work with
 * volatile variables.
 *
 *
 *  * `VolatileInt`: Provides semantics to work with valued integers.
 *  * `VolatileBool`: Provides semantics to work with valued booleans.
 *  * `VolatilePtr`: Provides semantics to work with pointers.
 *
 *
 * Implementing more specific volatile types such as `VolatileUint` and `VolatileU64` is
 * fairly easy.
 *
 *
 * This module uses [LLVM's volatile intrinsics][llvm].
 * [llvm]: http://llvm.org/docs/LangRef.html#volatile-memory-accesses
 */

use unstable::intrinsics;
use cast;
use option::{Option,Some, None};
use ptr;
use prelude::drop;

pub struct VolatileBool {
    priv v: uint
}

pub struct VolatileInt {
    priv v: int
}

/// Static initializers for the specific volatile types.
///
/// ```rust
/// static x: VolatileInt = INIT_VOLATILE_INT;
/// ```
pub static INIT_VOLATILE_BOOL: VolatileBool = VolatileBool { v: 0 };
pub static INIT_VOLATILE_INT: VolatileInt = VolatileInt { v: 0 };

impl VolatileInt {
    pub fn new(val: int) -> VolatileInt {
        VolatileInt { v: val }
    }

    #[inline]
    pub fn store(&mut self, val: int) {
        unsafe {
            intrinsics::volatile_store(&mut self.v, val);
        }
    }

    #[inline]
    pub fn load(&self) -> int {
        unsafe {
            intrinsics::volatile_load(&self.v as *int)
        }
    }
}


impl VolatileBool {
    pub fn new(val: bool) -> VolatileBool {
        VolatileBool { v: if val { 1 } else { 0} }
    }

    #[inline]
    pub fn store(&mut self, val: bool) {
        unsafe {
            intrinsics::volatile_store(&mut self.v, if val { 1 } else { 0 } );
        }
    }

    #[inline]
    pub fn load(&self) -> bool {
        unsafe {
            intrinsics::volatile_load(&self.v as *uint) != 0
        }
    }
}

pub struct VolatilePtr<T> {
    priv ptr: *mut T
}

/// VolatilePtr implementation that offers a safe wrapper around volatile types. This
/// particular implementation works with pointers; thus, it conforms to the ownership
/// semantics of Rust.
///
/// If you want to work with values, you may use the more specific implementations, such
/// as `VolatileInt` and `VolatileBool`.
impl<T> VolatilePtr<T> {
    /// Create a safe `VolatilePtr` based on the type given.
    pub fn new(ptr: ~T) -> VolatilePtr<T> {
        VolatilePtr {
            ptr: unsafe { cast::transmute(ptr) }
        }
    }

    /// Store a new pointer as a volatile variable.
    ///
    /// To make sure that we don't leak memory, we need to take ownership
    /// of a potential previous value. Then write the new contents, afterwhich
    /// we may drop the old contents.
    #[inline]
    pub fn store(&mut self, p: ~T) {
        unsafe {
            // Load the previous data. This will zero out the pointer, but we'll
            // have transfered the ownership of the previous data here.
            //
            // This function needs to drop the memory contents.
            let data = self.take();

            // Replace with the new data.
            let ptr: *mut T = cast::transmute(p);
            intrinsics::volatile_store(&mut self.ptr, ptr);

            // Drop the old data that we still have ownership of.
            match data {
                Some(t) => drop(t),
                None => { }
            }
        }
    }

    /// VolatilePtr conforms to the ownership semantics of Rust. Because VolatilePtr
    /// is storing a raw pointer, we must make sure that the value be transfered.
    ///
    /// The first operation is to load the memory, check if it's valid, zero out the
    /// memory contents, and return an owned pointer.
    ///
    /// This ensures that we transfer the ownership of the data, and that there aren't
    /// multiple copies available.
    #[inline]
    pub fn take(&mut self) -> Option<~T> {
        let ptr = unsafe { intrinsics::volatile_load(&self.ptr) };
        if ptr::is_null(ptr) { return None; }
        unsafe { intrinsics::volatile_store(&mut self.ptr, 0 as *mut T); }
        Some(unsafe { cast::transmute(ptr) })
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn volatile_ptr_new() {
        unsafe {
            let mut v = VolatilePtr::new(~5);
            assert_eq!(*v.take().unwrap(), 5);
        }
    }

    #[test]
    fn volatile_ptr_store_and_load() {
        unsafe {
            let mut v = VolatilePtr::new(~1);
            assert_eq!(*v.take().unwrap(), 1);

            v.store(~6);

            assert_eq!(*v.take().unwrap(), 6);
        }
    }

    #[test]
    fn volatile_int_load() {
        let vint = VolatileInt::new(16);
        assert_eq!(vint.load(), 16);
    }

    #[test]
    fn volatile_int_store() {
        let mut vint = VolatileInt::new(20);
        vint.store(10);
        assert_eq!(vint.load(), 10);
    }

    #[test]
    fn volatile_static_int() {
        static t: VolatileInt = INIT_VOLATILE_INT;
        static v: VolatileBool = INIT_VOLATILE_BOOL;
        assert_eq!(v.load(), false);
        assert_eq!(t.load(), 0);
    }

    #[test]
    fn volatile_bool_new_and_load() {
        let b = VolatileBool::new(true);
        assert_eq!(b.load(), true);
    }

    #[test]
    fn volatile_bool_store() {
        let mut b = VolatileBool::new(true);
        b.store(false);
        assert_eq!(b.load(), false);
    }
}
