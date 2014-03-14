// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations on managed box types

#[cfg(not(test))] use cmp::*;

/// Returns the refcount of a shared box (as just before calling this)
#[inline]
pub fn refcount<T>(t: @T) -> uint {
    use raw::Repr;
    unsafe { (*t.repr()).ref_count - 1 }
}

/// Determine if two shared boxes point to the same object
#[inline]
pub fn ptr_eq<T>(a: @T, b: @T) -> bool {
    &*a as *T == &*b as *T
}

#[cfg(not(test))]
impl<T:Eq> Eq for @T {
    #[inline]
    fn eq(&self, other: &@T) -> bool { *(*self) == *(*other) }
    #[inline]
    fn ne(&self, other: &@T) -> bool { *(*self) != *(*other) }
}

#[cfg(not(test))]
impl<T:Ord> Ord for @T {
    #[inline]
    fn lt(&self, other: &@T) -> bool { *(*self) < *(*other) }
    #[inline]
    fn le(&self, other: &@T) -> bool { *(*self) <= *(*other) }
    #[inline]
    fn ge(&self, other: &@T) -> bool { *(*self) >= *(*other) }
    #[inline]
    fn gt(&self, other: &@T) -> bool { *(*self) > *(*other) }
}

#[cfg(not(test))]
impl<T: TotalOrd> TotalOrd for @T {
    #[inline]
    fn cmp(&self, other: &@T) -> Ordering { (**self).cmp(*other) }
}

#[cfg(not(test))]
impl<T: TotalEq> TotalEq for @T {
    #[inline]
    fn equals(&self, other: &@T) -> bool { (**self).equals(*other) }
}

#[test]
fn test() {
    let x = @3;
    let y = @3;
    assert!((ptr_eq::<int>(x, x)));
    assert!((ptr_eq::<int>(y, y)));
    assert!((!ptr_eq::<int>(x, y)));
    assert!((!ptr_eq::<int>(y, x)));
}

#[test]
fn refcount_test() {
    use clone::Clone;

    let x = @3;
    assert_eq!(refcount(x), 1);
    let y = x.clone();
    assert_eq!(refcount(x), 2);
    assert_eq!(refcount(y), 2);
}
