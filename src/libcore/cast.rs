// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Unsafe casting functions

use mem;
use intrinsics;
use ptr::copy_nonoverlapping_memory;

/**
 * Transform a value of one type into a value of another type.
 * Both types must have the same size and alignment.
 *
 * # Example
 *
 * ```rust
 * use std::cast;
 *
 * let v: &[u8] = unsafe { cast::transmute("L") };
 * assert!(v == [76u8]);
 * ```
 */
#[inline]
pub unsafe fn transmute<T, U>(thing: T) -> U {
    intrinsics::transmute(thing)
}

/**
 * Move a thing into the void
 *
 * The forget function will take ownership of the provided value but neglect
 * to run any required cleanup or memory-management operations on it.
 */
#[inline]
pub unsafe fn forget<T>(thing: T) { intrinsics::forget(thing); }

/// Casts the value at `src` to U. The two types must have the same length.
#[inline]
pub unsafe fn transmute_copy<T, U>(src: &T) -> U {
    let mut dest: U = mem::uninit();
    let dest_ptr: *mut u8 = transmute(&mut dest);
    let src_ptr: *u8 = transmute(src);
    copy_nonoverlapping_memory(dest_ptr, src_ptr, mem::size_of::<U>());
    dest
}

/// Coerce an immutable reference to be mutable.
#[inline]
#[deprecated="casting &T to &mut T is undefined behaviour: use Cell<T>, RefCell<T> or Unsafe<T>"]
pub unsafe fn transmute_mut<'a,T>(ptr: &'a T) -> &'a mut T { transmute(ptr) }

/// Coerce a reference to have an arbitrary associated lifetime.
#[inline]
pub unsafe fn transmute_lifetime<'a,'b,T>(ptr: &'a T) -> &'b T {
    transmute(ptr)
}

/// Coerce an immutable reference to be mutable.
#[inline]
pub unsafe fn transmute_mut_unsafe<T>(ptr: *T) -> *mut T {
    transmute(ptr)
}

/// Coerce a mutable reference to have an arbitrary associated lifetime.
#[inline]
pub unsafe fn transmute_mut_lifetime<'a,'b,T>(ptr: &'a mut T) -> &'b mut T {
    transmute(ptr)
}

/// Transforms lifetime of the second pointer to match the first.
#[inline]
pub unsafe fn copy_lifetime<'a,S,T>(_ptr: &'a S, ptr: &T) -> &'a T {
    transmute_lifetime(ptr)
}

/// Transforms lifetime of the second pointer to match the first.
#[inline]
pub unsafe fn copy_mut_lifetime<'a,S,T>(_ptr: &'a mut S, ptr: &mut T) -> &'a mut T {
    transmute_mut_lifetime(ptr)
}

/// Transforms lifetime of the second pointer to match the first.
#[inline]
pub unsafe fn copy_lifetime_vec<'a,S,T>(_ptr: &'a [S], ptr: &T) -> &'a T {
    transmute_lifetime(ptr)
}


/****************************************************************************
 * Tests
 ****************************************************************************/

#[cfg(test)]
mod tests {
    use cast::transmute;
    use raw;
    use realstd::str::StrAllocating;

    #[test]
    fn test_transmute_copy() {
        assert_eq!(1u, unsafe { ::cast::transmute_copy(&1) });
    }

    #[test]
    fn test_transmute() {
        unsafe {
            let x = @100u8;
            let x: *raw::Box<u8> = transmute(x);
            assert!((*x).data == 100);
            let _x: @int = transmute(x);
        }
    }

    #[test]
    fn test_transmute2() {
        unsafe {
            assert_eq!(box [76u8], transmute("L".to_owned()));
        }
    }
}
