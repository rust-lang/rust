// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

The `ToBytes` and `IterBytes` traits

*/

use io;
use io::Writer;
use option::{None, Option, Some};
use str;

pub type Cb<'self> = &'self fn(buf: &[u8]) -> bool;

#[cfg(stage0)]
pub trait IterBytes {
    fn iter_bytes(&self, lsb0: bool, f: Cb);
}

/**
 * A trait to implement in order to make a type hashable;
 * This works in combination with the trait `Hash::Hash`, and
 * may in the future be merged with that trait or otherwise
 * modified when default methods and trait inheritence are
 * completed.
 */
#[cfg(not(stage0))]
pub trait IterBytes {
    /**
     * Call the provided callback `f` one or more times with
     * byte-slices that should be used when computing a hash
     * value or otherwise "flattening" the structure into
     * a sequence of bytes. The `lsb0` parameter conveys
     * whether the caller is asking for little-endian bytes
     * (`true`) or big-endian (`false`); this should only be
     * relevant in implementations that represent a single
     * multi-byte datum such as a 32 bit integer or 64 bit
     * floating-point value. It can be safely ignored for
     * larger structured types as they are usually processed
     * left-to-right in declaration order, regardless of
     * underlying memory endianness.
     */
    fn iter_bytes(&self, lsb0: bool, f: Cb) -> bool;
}

#[cfg(stage0)]
impl IterBytes for bool {
    #[inline(always)]
    fn iter_bytes(&self, _lsb0: bool, f: Cb) {
        f([
            *self as u8
        ]);
    }
}
#[cfg(not(stage0))]
impl IterBytes for bool {
    #[inline(always)]
    fn iter_bytes(&self, _lsb0: bool, f: Cb) -> bool {
        f([
            *self as u8
        ])
    }
}

#[cfg(stage0)]
impl IterBytes for u8 {
    #[inline(always)]
    fn iter_bytes(&self, _lsb0: bool, f: Cb) {
        f([
            *self
        ]);
    }
}
#[cfg(not(stage0))]
impl IterBytes for u8 {
    #[inline(always)]
    fn iter_bytes(&self, _lsb0: bool, f: Cb) -> bool {
        f([
            *self
        ])
    }
}

#[cfg(stage0)]
impl IterBytes for u16 {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) {
        if lsb0 {
            f([
                *self as u8,
                (*self >> 8) as u8
            ]);
        } else {
            f([
                (*self >> 8) as u8,
                *self as u8
            ]);
        }
    }
}
#[cfg(not(stage0))]
impl IterBytes for u16 {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) -> bool {
        if lsb0 {
            f([
                *self as u8,
                (*self >> 8) as u8
            ])
        } else {
            f([
                (*self >> 8) as u8,
                *self as u8
            ])
        }
    }
}

#[cfg(stage0)]
impl IterBytes for u32 {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) {
        if lsb0 {
            f([
                *self as u8,
                (*self >> 8) as u8,
                (*self >> 16) as u8,
                (*self >> 24) as u8,
            ]);
        } else {
            f([
                (*self >> 24) as u8,
                (*self >> 16) as u8,
                (*self >> 8) as u8,
                *self as u8
            ]);
        }
    }
}
#[cfg(not(stage0))]
impl IterBytes for u32 {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) -> bool {
        if lsb0 {
            f([
                *self as u8,
                (*self >> 8) as u8,
                (*self >> 16) as u8,
                (*self >> 24) as u8,
            ])
        } else {
            f([
                (*self >> 24) as u8,
                (*self >> 16) as u8,
                (*self >> 8) as u8,
                *self as u8
            ])
        }
    }
}

#[cfg(stage0)]
impl IterBytes for u64 {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) {
        if lsb0 {
            f([
                *self as u8,
                (*self >> 8) as u8,
                (*self >> 16) as u8,
                (*self >> 24) as u8,
                (*self >> 32) as u8,
                (*self >> 40) as u8,
                (*self >> 48) as u8,
                (*self >> 56) as u8
            ]);
        } else {
            f([
                (*self >> 56) as u8,
                (*self >> 48) as u8,
                (*self >> 40) as u8,
                (*self >> 32) as u8,
                (*self >> 24) as u8,
                (*self >> 16) as u8,
                (*self >> 8) as u8,
                *self as u8
            ]);
        }
    }
}
#[cfg(not(stage0))]
impl IterBytes for u64 {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) -> bool {
        if lsb0 {
            f([
                *self as u8,
                (*self >> 8) as u8,
                (*self >> 16) as u8,
                (*self >> 24) as u8,
                (*self >> 32) as u8,
                (*self >> 40) as u8,
                (*self >> 48) as u8,
                (*self >> 56) as u8
            ])
        } else {
            f([
                (*self >> 56) as u8,
                (*self >> 48) as u8,
                (*self >> 40) as u8,
                (*self >> 32) as u8,
                (*self >> 24) as u8,
                (*self >> 16) as u8,
                (*self >> 8) as u8,
                *self as u8
            ])
        }
    }
}

#[cfg(stage0)]
impl IterBytes for i8 {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) {
        (*self as u8).iter_bytes(lsb0, f)
    }
}
#[cfg(not(stage0))]
impl IterBytes for i8 {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) -> bool {
        (*self as u8).iter_bytes(lsb0, f)
    }
}

#[cfg(stage0)]
impl IterBytes for i16 {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) {
        (*self as u16).iter_bytes(lsb0, f)
    }
}
#[cfg(not(stage0))]
impl IterBytes for i16 {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) -> bool {
        (*self as u16).iter_bytes(lsb0, f)
    }
}

#[cfg(stage0)]
impl IterBytes for i32 {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) {
        (*self as u32).iter_bytes(lsb0, f)
    }
}
#[cfg(not(stage0))]
impl IterBytes for i32 {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) -> bool {
        (*self as u32).iter_bytes(lsb0, f)
    }
}

#[cfg(stage0)]
impl IterBytes for i64 {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) {
        (*self as u64).iter_bytes(lsb0, f)
    }
}
#[cfg(not(stage0))]
impl IterBytes for i64 {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) -> bool {
        (*self as u64).iter_bytes(lsb0, f)
    }
}

#[cfg(stage0)]
impl IterBytes for char {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) {
        (*self as u32).iter_bytes(lsb0, f)
    }
}
#[cfg(not(stage0))]
impl IterBytes for char {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) -> bool {
        (*self as u32).iter_bytes(lsb0, f)
    }
}

#[cfg(target_word_size = "32", stage0)]
impl IterBytes for uint {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) {
        (*self as u32).iter_bytes(lsb0, f)
    }
}
#[cfg(target_word_size = "32", not(stage0))]
impl IterBytes for uint {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) -> bool {
        (*self as u32).iter_bytes(lsb0, f)
    }
}

#[cfg(target_word_size = "64", stage0)]
impl IterBytes for uint {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) {
        (*self as u64).iter_bytes(lsb0, f)
    }
}
#[cfg(target_word_size = "64", not(stage0))]
impl IterBytes for uint {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) -> bool {
        (*self as u64).iter_bytes(lsb0, f)
    }
}

#[cfg(stage0)]
impl IterBytes for int {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) {
        (*self as uint).iter_bytes(lsb0, f)
    }
}
#[cfg(not(stage0))]
impl IterBytes for int {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) -> bool {
        (*self as uint).iter_bytes(lsb0, f)
    }
}

#[cfg(stage0)]
impl<'self,A:IterBytes> IterBytes for &'self [A] {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) {
        for (*self).each |elt| {
            do elt.iter_bytes(lsb0) |bytes| {
                f(bytes)
            }
        }
    }
}
#[cfg(not(stage0))]
impl<'self,A:IterBytes> IterBytes for &'self [A] {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) -> bool {
        self.each(|elt| elt.iter_bytes(lsb0, |b| f(b)))
    }
}

#[cfg(stage0)]
impl<A:IterBytes,B:IterBytes> IterBytes for (A,B) {
  #[inline(always)]
  fn iter_bytes(&self, lsb0: bool, f: Cb) {
    match *self {
      (ref a, ref b) => {
        iter_bytes_2(a, b, lsb0, f);
      }
    }
  }
}
#[cfg(not(stage0))]
impl<A:IterBytes,B:IterBytes> IterBytes for (A,B) {
  #[inline(always)]
  fn iter_bytes(&self, lsb0: bool, f: Cb) -> bool {
    match *self {
      (ref a, ref b) => { a.iter_bytes(lsb0, f) && b.iter_bytes(lsb0, f) }
    }
  }
}

#[cfg(stage0)]
impl<A:IterBytes,B:IterBytes,C:IterBytes> IterBytes for (A,B,C) {
  #[inline(always)]
  fn iter_bytes(&self, lsb0: bool, f: Cb) {
    match *self {
      (ref a, ref b, ref c) => {
        iter_bytes_3(a, b, c, lsb0, f);
      }
    }
  }
}
#[cfg(not(stage0))]
impl<A:IterBytes,B:IterBytes,C:IterBytes> IterBytes for (A,B,C) {
  #[inline(always)]
  fn iter_bytes(&self, lsb0: bool, f: Cb) -> bool {
    match *self {
      (ref a, ref b, ref c) => {
        a.iter_bytes(lsb0, f) && b.iter_bytes(lsb0, f) && c.iter_bytes(lsb0, f)
      }
    }
  }
}

// Move this to vec, probably.
fn borrow<'x,A>(a: &'x [A]) -> &'x [A] {
    a
}

#[cfg(stage0)]
impl<A:IterBytes> IterBytes for ~[A] {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) {
        borrow(*self).iter_bytes(lsb0, f)
    }
}
#[cfg(not(stage0))]
impl<A:IterBytes> IterBytes for ~[A] {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) -> bool {
        borrow(*self).iter_bytes(lsb0, f)
    }
}

#[cfg(stage0)]
impl<A:IterBytes> IterBytes for @[A] {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) {
        borrow(*self).iter_bytes(lsb0, f)
    }
}
#[cfg(not(stage0))]
impl<A:IterBytes> IterBytes for @[A] {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) -> bool {
        borrow(*self).iter_bytes(lsb0, f)
    }
}

// NOTE: remove all of these after a snapshot, the new for-loop iteration
//       protocol makes these unnecessary.

#[cfg(stage0)]
pub fn iter_bytes_2<A:IterBytes,B:IterBytes>(a: &A, b: &B,
                                            lsb0: bool, z: Cb) {
    let mut flag = true;
    a.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
    if !flag { return; }
    b.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
}
#[cfg(not(stage0))]
pub fn iter_bytes_2<A:IterBytes,B:IterBytes>(a: &A, b: &B,
                                             lsb0: bool, z: Cb) -> bool {
    a.iter_bytes(lsb0, z) && b.iter_bytes(lsb0, z)
}

#[cfg(stage0)]
pub fn iter_bytes_3<A: IterBytes,
                    B: IterBytes,
                    C: IterBytes>(a: &A, b: &B, c: &C,
                                  lsb0: bool, z: Cb) {
    let mut flag = true;
    a.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
    if !flag { return; }
    b.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
    if !flag { return; }
    c.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
}
#[cfg(not(stage0))]
pub fn iter_bytes_3<A: IterBytes,
                    B: IterBytes,
                    C: IterBytes>(a: &A, b: &B, c: &C, lsb0: bool, z: Cb) -> bool {
    a.iter_bytes(lsb0, z) && b.iter_bytes(lsb0, z) && c.iter_bytes(lsb0, z)
}

#[cfg(stage0)]
pub fn iter_bytes_4<A: IterBytes,
                B: IterBytes,
                C: IterBytes,
                D: IterBytes>(a: &A, b: &B, c: &C,
                              d: &D,
                              lsb0: bool, z: Cb) {
    let mut flag = true;
    a.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
    if !flag { return; }
    b.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
    if !flag { return; }
    c.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
    if !flag { return; }
    d.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
}
#[cfg(not(stage0))]
pub fn iter_bytes_4<A: IterBytes,
                B: IterBytes,
                C: IterBytes,
                D: IterBytes>(a: &A, b: &B, c: &C,
                              d: &D,
                              lsb0: bool, z: Cb) -> bool {
    a.iter_bytes(lsb0, z) && b.iter_bytes(lsb0, z) && c.iter_bytes(lsb0, z) &&
        d.iter_bytes(lsb0, z)
}

#[cfg(stage0)]
pub fn iter_bytes_5<A: IterBytes,
                B: IterBytes,
                C: IterBytes,
                D: IterBytes,
                E: IterBytes>(a: &A, b: &B, c: &C,
                              d: &D, e: &E,
                              lsb0: bool, z: Cb) {
    let mut flag = true;
    a.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
    if !flag { return; }
    b.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
    if !flag { return; }
    c.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
    if !flag { return; }
    d.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
    if !flag { return; }
    e.iter_bytes(lsb0, |bytes| {flag = z(bytes); flag});
}
#[cfg(not(stage0))]
pub fn iter_bytes_5<A: IterBytes,
                B: IterBytes,
                C: IterBytes,
                D: IterBytes,
                E: IterBytes>(a: &A, b: &B, c: &C,
                              d: &D, e: &E,
                              lsb0: bool, z: Cb) -> bool {
    a.iter_bytes(lsb0, z) && b.iter_bytes(lsb0, z) && c.iter_bytes(lsb0, z) &&
        d.iter_bytes(lsb0, z) && e.iter_bytes(lsb0, z)
}

#[cfg(stage0)]
impl<'self> IterBytes for &'self str {
    #[inline(always)]
    fn iter_bytes(&self, _lsb0: bool, f: Cb) {
        do str::byte_slice(*self) |bytes| {
            f(bytes);
        }
    }
}
#[cfg(not(stage0))]
impl<'self> IterBytes for &'self str {
    #[inline(always)]
    fn iter_bytes(&self, _lsb0: bool, f: Cb) -> bool {
        do str::byte_slice(*self) |bytes| {
            f(bytes)
        }
    }
}

#[cfg(stage0)]
impl IterBytes for ~str {
    #[inline(always)]
    fn iter_bytes(&self, _lsb0: bool, f: Cb) {
        do str::byte_slice(*self) |bytes| {
            f(bytes);
        }
    }
}
#[cfg(not(stage0))]
impl IterBytes for ~str {
    #[inline(always)]
    fn iter_bytes(&self, _lsb0: bool, f: Cb) -> bool {
        do str::byte_slice(*self) |bytes| {
            f(bytes)
        }
    }
}

#[cfg(stage0)]
impl IterBytes for @str {
    #[inline(always)]
    fn iter_bytes(&self, _lsb0: bool, f: Cb) {
        do str::byte_slice(*self) |bytes| {
            f(bytes);
        }
    }
}
#[cfg(not(stage0))]
impl IterBytes for @str {
    #[inline(always)]
    fn iter_bytes(&self, _lsb0: bool, f: Cb) -> bool {
        do str::byte_slice(*self) |bytes| {
            f(bytes)
        }
    }
}

#[cfg(stage0)]
impl<A:IterBytes> IterBytes for Option<A> {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) {
        match *self {
          Some(ref a) => iter_bytes_2(&0u8, a, lsb0, f),
          None => 1u8.iter_bytes(lsb0, f)
        }
    }
}
#[cfg(not(stage0))]
impl<A:IterBytes> IterBytes for Option<A> {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) -> bool {
        match *self {
          Some(ref a) => 0u8.iter_bytes(lsb0, f) && a.iter_bytes(lsb0, f),
          None => 1u8.iter_bytes(lsb0, f)
        }
    }
}

#[cfg(stage0)]
impl<'self,A:IterBytes> IterBytes for &'self A {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) {
        (**self).iter_bytes(lsb0, f);
    }
}
#[cfg(not(stage0))]
impl<'self,A:IterBytes> IterBytes for &'self A {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) -> bool {
        (**self).iter_bytes(lsb0, f)
    }
}

#[cfg(stage0)]
impl<A:IterBytes> IterBytes for @A {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) {
        (**self).iter_bytes(lsb0, f);
    }
}
#[cfg(not(stage0))]
impl<A:IterBytes> IterBytes for @A {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) -> bool {
        (**self).iter_bytes(lsb0, f)
    }
}

#[cfg(stage0)]
impl<A:IterBytes> IterBytes for ~A {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) {
        (**self).iter_bytes(lsb0, f);
    }
}
#[cfg(not(stage0))]
impl<A:IterBytes> IterBytes for ~A {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) -> bool {
        (**self).iter_bytes(lsb0, f)
    }
}

// NB: raw-pointer IterBytes does _not_ dereference
// to the target; it just gives you the pointer-bytes.
#[cfg(stage0)]
impl<A> IterBytes for *const A {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) {
        (*self as uint).iter_bytes(lsb0, f);
    }
}
// NB: raw-pointer IterBytes does _not_ dereference
// to the target; it just gives you the pointer-bytes.
#[cfg(not(stage0))]
impl<A> IterBytes for *const A {
    #[inline(always)]
    fn iter_bytes(&self, lsb0: bool, f: Cb) -> bool {
        (*self as uint).iter_bytes(lsb0, f)
    }
}

pub trait ToBytes {
    fn to_bytes(&self, lsb0: bool) -> ~[u8];
}

impl<A:IterBytes> ToBytes for A {
    fn to_bytes(&self, lsb0: bool) -> ~[u8] {
        do io::with_bytes_writer |wr| {
            for self.iter_bytes(lsb0) |bytes| {
                wr.write(bytes)
            }
        }
    }
}
