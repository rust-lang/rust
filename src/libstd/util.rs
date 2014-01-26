// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Miscellaneous helpers for common patterns

use cast;
use ptr;
use unstable::intrinsics;

/// The identity function.
#[inline]
pub fn id<T>(x: T) -> T { x }

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
        let x_raw: *mut T = x;
        let y_raw: *mut T = y;
        ptr::copy_nonoverlapping_memory(t, x_raw, 1);
        ptr::copy_nonoverlapping_memory(x, y_raw, 1);
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


#[cfg(test)]
mod tests {
    use super::*;
    use prelude::*;
    use mem::size_of;

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
        let mut x = Some(~"test");
        let y = replace(&mut x, None);
        assert!(x.is_none());
        assert!(y.is_some());
    }
}

/// Completely miscellaneous language-construct benchmarks.
#[cfg(test)]
mod bench {

    use extra::test::BenchHarness;
    use option::{Some,None};

    // Static/dynamic method dispatch

    struct Struct {
        field: int
    }

    trait Trait {
        fn method(&self) -> int;
    }

    impl Trait for Struct {
        fn method(&self) -> int {
            self.field
        }
    }

    #[bench]
    fn trait_vtable_method_call(bh: &mut BenchHarness) {
        let s = Struct { field: 10 };
        let t = &s as &Trait;
        bh.iter(|| {
            t.method();
        });
    }

    #[bench]
    fn trait_static_method_call(bh: &mut BenchHarness) {
        let s = Struct { field: 10 };
        bh.iter(|| {
            s.method();
        });
    }

    // Overhead of various match forms

    #[bench]
    fn match_option_some(bh: &mut BenchHarness) {
        let x = Some(10);
        bh.iter(|| {
            let _q = match x {
                Some(y) => y,
                None => 11
            };
        });
    }

    #[bench]
    fn match_vec_pattern(bh: &mut BenchHarness) {
        let x = [1,2,3,4,5,6];
        bh.iter(|| {
            let _q = match x {
                [1,2,3,..] => 10,
                _ => 11
            };
        });
    }
}
