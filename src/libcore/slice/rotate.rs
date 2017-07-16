// Copyright 2012-2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use cmp;
use mem;
use ptr;

/// Rotation is much faster if it has access to a little bit of memory. This
/// union provides a RawVec-like interface, but to a fixed-size stack buffer.
#[allow(unions_with_drop_fields)]
union RawArray<T> {
    /// Ensure this is appropriately aligned for T, and is big
    /// enough for two elements even if T is enormous.
    typed: [T; 2],
    /// For normally-sized types, especially things like u8, having more
    /// than 2 in the buffer is necessary for usefulness, so pad it out
    /// enough to be helpful, but not so big as to risk overflow.
    _extra: [usize; 32],
}

impl<T> RawArray<T> {
    fn new() -> Self {
        unsafe { mem::uninitialized() }
    }
    fn ptr(&self) -> *mut T {
        unsafe { &self.typed as *const T as *mut T }
    }
    fn cap() -> usize {
        if mem::size_of::<T>() == 0 {
            usize::max_value()
        } else {
            mem::size_of::<Self>() / mem::size_of::<T>()
        }
    }
}

/// Rotates the range `[mid-left, mid+right)` such that the element at `mid`
/// becomes the first element.  Equivalently, rotates the range `left`
/// elements to the left or `right` elements to the right.
///
/// # Safety
///
/// The specified range must be valid for reading and writing.
/// The type `T` must have non-zero size.
///
/// # Algorithm
///
/// For longer rotations, swap the left-most `delta = min(left, right)`
/// elements with the right-most `delta` elements.  LLVM vectorizes this,
/// which is profitable as we only reach this step for a "large enough"
/// rotation.  Doing this puts `delta` elements on the larger side into the
/// correct position, leaving a smaller rotate problem.  Demonstration:
///
/// ```text
/// [ 6 7 8 9 10 11 12 13 . 1 2 3 4 5 ]
/// 1 2 3 4 5 [ 11 12 13 . 6 7 8 9 10 ]
/// 1 2 3 4 5 [ 8 9 10 . 6 7 ] 11 12 13
/// 1 2 3 4 5 6 7 [ 10 . 8 9 ] 11 12 13
/// 1 2 3 4 5 6 7 [ 9 . 8 ] 10 11 12 13
/// 1 2 3 4 5 6 7 8 [ . ] 9 10 11 12 13
/// ```
///
/// Once the rotation is small enough, copy some elements into a stack
/// buffer, `memmove` the others, and move the ones back from the buffer.
pub unsafe fn ptr_rotate<T>(mut left: usize, mid: *mut T, mut right: usize) {
    loop {
        let delta = cmp::min(left, right);
        if delta <= RawArray::<T>::cap() {
            break;
        }

        ptr::swap_nonoverlapping(
            mid.offset(-(left as isize)),
            mid.offset((right-delta) as isize),
            delta);

        if left <= right {
            right -= delta;
        } else {
            left -= delta;
        }
    }

    let rawarray = RawArray::new();
    let buf = rawarray.ptr();

    let dim = mid.offset(-(left as isize)).offset(right as isize);
    if left <= right {
        ptr::copy_nonoverlapping(mid.offset(-(left as isize)), buf, left);
        ptr::copy(mid, mid.offset(-(left as isize)), right);
        ptr::copy_nonoverlapping(buf, dim, left);
    }
    else {
        ptr::copy_nonoverlapping(mid, buf, right);
        ptr::copy(mid.offset(-(left as isize)), dim, left);
        ptr::copy_nonoverlapping(buf, mid.offset(-(left as isize)), right);
    }
}
