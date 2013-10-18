// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Unsafe casting functions

use ptr::RawPtr;
use mem;
use unstable::intrinsics;

/// Casts the value at `src` to U. The two types must have the same length.
#[cfg(target_word_size = "32")]
#[inline]
pub unsafe fn transmute_copy<T, U>(src: &T) -> U {
    let mut dest: U = intrinsics::uninit();
    let dest_ptr: *mut u8 = transmute(&mut dest);
    let src_ptr: *u8 = transmute(src);
    intrinsics::memcpy32(dest_ptr, src_ptr, mem::size_of::<U>() as u32);
    dest
}

/// Casts the value at `src` to U. The two types must have the same length.
#[cfg(target_word_size = "64")]
#[inline]
pub unsafe fn transmute_copy<T, U>(src: &T) -> U {
    let mut dest: U = intrinsics::uninit();
    let dest_ptr: *mut u8 = transmute(&mut dest);
    let src_ptr: *u8 = transmute(src);
    intrinsics::memcpy64(dest_ptr, src_ptr, mem::size_of::<U>() as u64);
    dest
}

/**
 * Forces a copy of a value, even if that value is considered noncopyable.
 */
#[inline]
pub unsafe fn unsafe_copy<T>(thing: &T) -> T {
    transmute_copy(thing)
}

/**
 * Move a thing into the void
 *
 * The forget function will take ownership of the provided value but neglect
 * to run any required cleanup or memory-management operations on it. This
 * can be used for various acts of magick.
 */
#[inline]
pub unsafe fn forget<T>(thing: T) { intrinsics::forget(thing); }

/**
 * Force-increment the reference count on a shared box. If used
 * carelessly, this can leak the box.
 */
#[inline]
pub unsafe fn bump_box_refcount<T>(t: @T) { forget(t); }

/**
 * Transform a value of one type into a value of another type.
 * Both types must have the same size and alignment.
 *
 * # Example
 *
 * ```rust
 * let v: &[u8] = transmute("L");
 * assert!(v == [76u8]);
 * ```
 */
#[inline]
pub unsafe fn transmute<L, G>(thing: L) -> G {
    intrinsics::transmute(thing)
}

/// Coerce an immutable reference to be mutable.
#[inline]
pub unsafe fn transmute_mut<'a,T>(ptr: &'a T) -> &'a mut T { transmute(ptr) }

/// Coerce a mutable reference to be immutable.
#[inline]
pub unsafe fn transmute_immut<'a,T>(ptr: &'a mut T) -> &'a T {
    transmute(ptr)
}

/// Coerce a borrowed pointer to have an arbitrary associated region.
#[inline]
pub unsafe fn transmute_region<'a,'b,T>(ptr: &'a T) -> &'b T {
    transmute(ptr)
}

/// Coerce an immutable reference to be mutable.
#[inline]
pub unsafe fn transmute_mut_unsafe<T,P:RawPtr<T>>(ptr: P) -> *mut T {
    transmute(ptr)
}

/// Coerce an immutable reference to be mutable.
#[inline]
pub unsafe fn transmute_immut_unsafe<T,P:RawPtr<T>>(ptr: P) -> *T {
    transmute(ptr)
}

/// Coerce a borrowed mutable pointer to have an arbitrary associated region.
#[inline]
pub unsafe fn transmute_mut_region<'a,'b,T>(ptr: &'a mut T) -> &'b mut T {
    transmute(ptr)
}

/// Transforms lifetime of the second pointer to match the first.
#[inline]
pub unsafe fn copy_lifetime<'a,S,T>(_ptr: &'a S, ptr: &T) -> &'a T {
    transmute_region(ptr)
}

/// Transforms lifetime of the second pointer to match the first.
#[inline]
pub unsafe fn copy_mut_lifetime<'a,S,T>(_ptr: &'a mut S, ptr: &mut T) -> &'a mut T {
    transmute_mut_region(ptr)
}

/// Transforms lifetime of the second pointer to match the first.
#[inline]
pub unsafe fn copy_lifetime_vec<'a,S,T>(_ptr: &'a [S], ptr: &T) -> &'a T {
    transmute_region(ptr)
}


/****************************************************************************
 * Tests
 ****************************************************************************/

#[cfg(test)]
mod tests {
    use cast::{bump_box_refcount, transmute};
    use unstable::raw;

    #[test]
    fn test_transmute_copy() {
        assert_eq!(1u, unsafe { ::cast::transmute_copy(&1) });
    }

    #[test]
    fn test_bump_box_refcount() {
        unsafe {
            let box = @~"box box box";       // refcount 1
            bump_box_refcount(box);         // refcount 2
            let ptr: *int = transmute(box); // refcount 2
            let _box1: @~str = ::cast::transmute_copy(&ptr);
            let _box2: @~str = ::cast::transmute_copy(&ptr);
            assert!(*_box1 == ~"box box box");
            assert!(*_box2 == ~"box box box");
            // Will destroy _box1 and _box2. Without the bump, this would
            // use-after-free. With too many bumps, it would leak.
        }
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
            assert_eq!(~[76u8], transmute(~"L"));
        }
    }
}
