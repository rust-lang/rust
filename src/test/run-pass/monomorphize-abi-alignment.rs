// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * On x86_64-linux-gnu and possibly other platforms, structs get 8-byte "preferred" alignment,
 * but their "ABI" alignment (i.e., what actually matters for data layout) is the largest alignment
 * of any field.  (Also, u64 has 8-byte ABI alignment; this is not always true).
 *
 * On such platforms, if monomorphize uses the "preferred" alignment, then it will unify
 * `A` and `B`, even though `S<A>` and `S<B>` have the field `t` at different offsets,
 * and apply the wrong instance of the method `unwrap`.
 */

struct S<T> { i:u8, t:T }
impl<T> S<T> { fn unwrap(self) -> T { self.t } }
#[deriving(Eq)]
struct A((u32, u32));
#[deriving(Eq)]
struct B(u64);

pub fn main() {
    static Ca: S<A> = S { i: 0, t: A((13, 104)) };
    static Cb: S<B> = S { i: 0, t: B(31337) };
    assert_eq!(Ca.unwrap(), A((13, 104)));
    assert_eq!(Cb.unwrap(), B(31337));
}
