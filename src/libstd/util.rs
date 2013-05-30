// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

Miscellaneous helpers for common patterns.

*/

use prelude::*;
use unstable::intrinsics;

/// The identity function.
#[inline(always)]
pub fn id<T>(x: T) -> T { x }

/// Ignores a value.
#[inline(always)]
pub fn ignore<T>(_x: T) { }

/// Sets `*ptr` to `new_value`, invokes `op()`, and then restores the
/// original value of `*ptr`.
///
/// NB: This function accepts `@mut T` and not `&mut T` to avoid
/// an obvious borrowck hazard. Typically passing in `&mut T` will
/// cause borrow check errors because it freezes whatever location
/// that `&mut T` is stored in (either statically or dynamically).
#[inline(always)]
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
#[inline(always)]
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
 * Swap the values at two mutable locations of the same type, without
 * deinitialising or copying either one.
 */
#[inline]
pub unsafe fn swap_ptr<T>(x: *mut T, y: *mut T) {
    // Give ourselves some scratch space to work with
    let mut tmp: T = intrinsics::uninit();
    let t: *mut T = &mut tmp;

    // Perform the swap
    ptr::copy_memory(t, x, 1);
    ptr::copy_memory(x, y, 1);
    ptr::copy_memory(y, t, 1);

    // y and t now point to the same thing, but we need to completely forget `tmp`
    // because it's no longer relevant.
    cast::forget(tmp);
}

/**
 * Replace the value at a mutable location with a new one, returning the old
 * value, without deinitialising or copying either one.
 */
#[inline(always)]
pub fn replace<T>(dest: &mut T, mut src: T) -> T {
    swap(dest, &mut src);
    src
}

/**
 * Replace the value at a mutable location with a new one, returning the old
 * value, without deinitialising or copying either one.
 */
#[inline(always)]
pub unsafe fn replace_ptr<T>(dest: *mut T, mut src: T) -> T {
    swap_ptr(dest, ptr::to_mut_unsafe_ptr(&mut src));
    src
}

/// A non-copyable dummy type.
pub struct NonCopyable {
    i: (),
}

impl Drop for NonCopyable {
    fn finalize(&self) { }
}

pub fn NonCopyable() -> NonCopyable { NonCopyable { i: () } }


/// A type with no inhabitants
pub enum Void { }

pub impl Void {
    /// A utility function for ignoring this uninhabited type
    fn uninhabited(self) -> ! {
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
    use option::{None, Some};
    use util::{Void, NonCopyable, id, replace, swap};
    use either::{Either, Left, Right};

    #[test]
    pub fn identity_crisis() {
        // Writing a test for the identity function. How did it come to this?
        let x = ~[(5, false)];
        //FIXME #3387 assert!(x.eq(id(copy x)));
        let y = copy x;
        assert!(x.eq(&id(y)));
    }
    #[test]
    pub fn test_swap() {
        let mut x = 31337;
        let mut y = 42;
        swap(&mut x, &mut y);
        assert_eq!(x, 42);
        assert_eq!(y, 31337);
    }
    #[test]
    pub fn test_replace() {
        let mut x = Some(NonCopyable());
        let y = replace(&mut x, None);
        assert!(x.is_none());
        assert!(y.is_some());
    }
    #[test]
    pub fn test_uninhabited() {
        let could_only_be_coin : Either <Void, ()> = Right (());
        match could_only_be_coin {
            Right (coin) => coin,
            Left (is_void) => is_void.uninhabited ()
        }
    }
}
