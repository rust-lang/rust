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

pub mod rusti {
    #[abi = "rust-intrinsic"]
    #[link_name = "rusti"]
    pub extern "rust-intrinsic" {
        fn forget<T>(+x: T);
        fn reinterpret_cast<T, U>(&&e: T) -> U;
    }
}

/// Casts the value at `src` to U. The two types must have the same length.
#[inline(always)]
pub unsafe fn reinterpret_cast<T, U>(src: &T) -> U {
    rusti::reinterpret_cast(*src)
}

/**
 * Move a thing into the void
 *
 * The forget function will take ownership of the provided value but neglect
 * to run any required cleanup or memory-management operations on it. This
 * can be used for various acts of magick, particularly when using
 * reinterpret_cast on pointer types.
 */
#[inline(always)]
pub unsafe fn forget<T>(thing: T) { rusti::forget(thing); }

/**
 * Force-increment the reference count on a shared box. If used
 * carelessly, this can leak the box. Use this in conjunction with transmute
 * and/or reinterpret_cast when such calls would otherwise scramble a box's
 * reference count
 */
pub unsafe fn bump_box_refcount<T>(t: @T) { forget(t); }

/**
 * Transform a value of one type into a value of another type.
 * Both types must have the same size and alignment.
 *
 * # Example
 *
 *     assert!(transmute("L") == ~[76u8, 0u8]);
 */
#[inline(always)]
pub unsafe fn transmute<L, G>(thing: L) -> G {
    let newthing: G = reinterpret_cast(&thing);
    forget(thing);
    newthing
}

/// Coerce an immutable reference to be mutable.
#[inline(always)]
pub unsafe fn transmute_mut<'a,T>(ptr: &'a T) -> &'a mut T { transmute(ptr) }

/// Coerce a mutable reference to be immutable.
#[inline(always)]
pub unsafe fn transmute_immut<'a,T>(ptr: &'a mut T) -> &'a T {
    transmute(ptr)
}

/// Coerce a borrowed pointer to have an arbitrary associated region.
#[inline(always)]
pub unsafe fn transmute_region<'a,'b,T>(ptr: &'a T) -> &'b T {
    transmute(ptr)
}

/// Coerce an immutable reference to be mutable.
#[inline(always)]
pub unsafe fn transmute_mut_unsafe<T>(ptr: *const T) -> *mut T {
    transmute(ptr)
}

/// Coerce an immutable reference to be mutable.
#[inline(always)]
pub unsafe fn transmute_immut_unsafe<T>(ptr: *const T) -> *T {
    transmute(ptr)
}

/// Coerce a borrowed mutable pointer to have an arbitrary associated region.
#[inline(always)]
pub unsafe fn transmute_mut_region<'a,'b,T>(ptr: &'a mut T) -> &'b mut T {
    transmute(ptr)
}

/// Transforms lifetime of the second pointer to match the first.
#[inline(always)]
pub unsafe fn copy_lifetime<'a,S,T>(_ptr: &'a S, ptr: &T) -> &'a T {
    transmute_region(ptr)
}

/// Transforms lifetime of the second pointer to match the first.
#[inline(always)]
pub unsafe fn copy_lifetime_vec<'a,S,T>(_ptr: &'a [S], ptr: &T) -> &'a T {
    transmute_region(ptr)
}


/****************************************************************************
 * Tests
 ****************************************************************************/

#[cfg(test)]
mod tests {
    use cast::{bump_box_refcount, reinterpret_cast, transmute};

    #[test]
    fn test_reinterpret_cast() {
        assert!(1u == unsafe { reinterpret_cast(&1) });
    }

    #[test]
    fn test_bump_box_refcount() {
        unsafe {
            let box = @~"box box box";       // refcount 1
            bump_box_refcount(box);         // refcount 2
            let ptr: *int = transmute(box); // refcount 2
            let _box1: @~str = reinterpret_cast(&ptr);
            let _box2: @~str = reinterpret_cast(&ptr);
            assert!(*_box1 == ~"box box box");
            assert!(*_box2 == ~"box box box");
            // Will destroy _box1 and _box2. Without the bump, this would
            // use-after-free. With too many bumps, it would leak.
        }
    }

    #[test]
    fn test_transmute() {
        use managed::raw::BoxRepr;
        unsafe {
            let x = @100u8;
            let x: *BoxRepr = transmute(x);
            assert!((*x).data == 100);
            let _x: @int = transmute(x);
        }
    }

    #[test]
    fn test_transmute2() {
        unsafe {
            assert!(~[76u8, 0u8] == transmute(~"L"));
        }
    }
}
