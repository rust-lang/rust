// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Miscellaneous helpers for common patterns.

use cast;
use ptr;
use prelude::*;
use unstable::intrinsics;

/// The identity function.
#[inline]
pub fn id<T>(x: T) -> T { x }

/// Ignores a value.
#[inline]
pub fn ignore<T>(_x: T) { }

/// Sets `*ptr` to `new_value`, invokes `op()`, and then restores the
/// original value of `*ptr`.
///
/// NB: This function accepts `@mut T` and not `&mut T` to avoid
/// an obvious borrowck hazard. Typically passing in `&mut T` will
/// cause borrow check errors because it freezes whatever location
/// that `&mut T` is stored in (either statically or dynamically).
#[inline]
pub fn with<T,R>(
    ptr: @mut T,
    value: T,
    op: &fn() -> R) -> R
{
    let prev = replace(ptr, value);
    let result = op();
    *ptr = prev;
    return result;
}

/**
 * Swap the values at two mutable locations of the same type, without
 * deinitialising or copying either one.
 */
#[inline]
pub fn swap<T>(x: &mut T, y: &mut T) {
    unsafe {
        // Give ourselves some scratch space to work with
        let mut tmp: T = intrinsics::uninit();
        let t: *mut T = &mut tmp;

        // Perform the swap, `&mut` pointers never alias
        ptr::copy_nonoverlapping_memory(t, x, 1);
        ptr::copy_nonoverlapping_memory(x, y, 1);
        ptr::copy_nonoverlapping_memory(y, t, 1);

        // y and t now point to the same thing, but we need to completely forget `tmp`
        // because it's no longer relevant.
        cast::forget(tmp);
    }
}

/**
 * Replace the value at a mutable location with a new one, returning the old
 * value, without deinitialising or copying either one.
 */
#[inline]
pub fn replace<T>(dest: &mut T, mut src: T) -> T {
    swap(dest, &mut src);
    src
}

/**
 * Trasform a value without copying or deinitializing it.
 *
 * Replaces illegal statements of the form:
 * x = trans(x);
 */
#[inline]
pub fn reset<T>(val: &mut T, trans: &fn(T) -> T) {
    unsafe {
        let mut tmp: T = intrinsics::uninit();
        swap(val, &mut tmp);

        let nothing = replace(val, trans(tmp));
        cast::forget(nothing);
    }
}

/**
 * Combine two values and replace the original without copying or deinitializing
 * either one.
 *
 * Replaces illegal statements of the form:
 * x = comb(x, y);
 */
#[inline]
pub fn pack<T, U>(val: &mut T, param: U, comb: &fn(T, U) -> T) {
    unsafe {
        let mut tmp: T = intrinsics::uninit();

        swap(val, &mut tmp);

        // Perform the combination into a new T
        let new = comb(tmp, param);

        let nothing = replace(val, new);

        // Don't let the tmp variable get dropped
        cast::forget(nothing);
    }
}

/**
 * Split one value into two and replace the original, returning the remainder.
 *
 * Replaces illegal expressions of the form:
 * (x, y) = split(x);
 * y
 */
#[inline]
pub fn unpack<T, U>(val: &mut T, split: &fn(T) -> (T, U)) -> U {
    unsafe {
        let mut tmp: T = intrinsics::uninit();

        swap(val, &mut tmp);

        let (new, result) = split(tmp);

        let nothing = replace(val, new);

        // Don't let the tmp variable get dropped
        cast::forget(nothing);

        // Return the unpacked value
        result
    }
}

/// A non-copyable dummy type.
#[deriving(Eq, TotalEq, Ord, TotalOrd)]
#[unsafe_no_drop_flag]
pub struct NonCopyable;

impl Drop for NonCopyable {
    fn drop(&self) { }
}

/// A type with no inhabitants
pub enum Void { }

impl Void {
    /// A utility function for ignoring this uninhabited type
    pub fn uninhabited(self) -> ! {
        match self {
            // Nothing to match on
        }
    }
}


/**
A utility function for indicating unreachable code. It will fail if
executed. This is occasionally useful to put after loops that never
terminate normally, but instead directly return from a function.

# Example

~~~ {.rust}
fn choose_weighted_item(v: &[Item]) -> Item {
    assert!(!v.is_empty());
    let mut so_far = 0u;
    for v.each |item| {
        so_far += item.weight;
        if so_far > 100 {
            return item;
        }
    }
    // The above loop always returns, so we must hint to the
    // type checker that it isn't possible to get down here
    util::unreachable();
}
~~~

*/
pub fn unreachable() -> ! {
    fail!("internal error: entered unreachable code");
}

#[cfg(test)]
mod tests {
    use super::*;

    use clone::Clone;
    use option::{None, Some};
    use either::{Either, Left, Right};
    use sys::size_of;
    use kinds::Drop;

    #[test]
    fn identity_crisis() {
        // Writing a test for the identity function. How did it come to this?
        let x = ~[(5, false)];
        //FIXME #3387 assert!(x.eq(id(x.clone())));
        let y = x.clone();
        assert!(x.eq(&id(y)));
    }

    #[test]
    fn test_swap() {
        let mut x = 31337;
        let mut y = 42;
        swap(&mut x, &mut y);
        assert_eq!(x, 42);
        assert_eq!(y, 31337);
    }

    #[test]
    fn test_replace() {
        let mut x = Some(NonCopyable);
        let y = replace(&mut x, None);
        assert!(x.is_none());
        assert!(y.is_some());
    }

    #[test]
    fn test_reset() {
        let mut val = ~5;
        do reset(&mut val) |v| {
            assert_eq!(*v, 5);
            ~3
        }
        assert_eq!(*v, 3);
    }

    #[test]
    fn test_pack() {
        let mut val = ~7;
        do pack(&mut val, 3) |v, i| {
            assert_eq!(*v, 7);
            assert_eq!(i, 3);
            ~(*v + i)
        }
        assert_eq!(*val, 10);
    }

    #[test]
    fn test_unpack() {
        let mut val = ~11;
        let m = do unpack(&val) |v| {
            assert_eq!(*v, 11);
            (v / 5, v % 5)
        };
        assert_eq!(val, 2);
        assert_eq!(m, 1);
    }

    #[test]
    fn test_uninhabited() {
        let could_only_be_coin : Either <Void, ()> = Right (());
        match could_only_be_coin {
            Right (coin) => coin,
            Left (is_void) => is_void.uninhabited ()
        }
    }

    #[test]
    fn test_noncopyable() {
        assert_eq!(size_of::<NonCopyable>(), 0);

        // verify that `#[unsafe_no_drop_flag]` works as intended on a zero-size struct

        static mut did_run: bool = false;

        struct Foo { five: int }

        impl Drop for Foo {
            fn drop(&self) {
                assert_eq!(self.five, 5);
                unsafe {
                    did_run = true;
                }
            }
        }

        {
            let _a = (NonCopyable, Foo { five: 5 }, NonCopyable);
        }

        unsafe { assert_eq!(did_run, true); }
    }
}
