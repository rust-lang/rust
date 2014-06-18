// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[deriving(PartialEq, Show)]
struct Partial<T> { x: T, y: T }

#[deriving(PartialEq, Show)]
struct S { val: int }
impl S { fn new(v: int) -> S { S { val: v } } }
impl Drop for S { fn drop(&mut self) { } }

type Two<T> = (Partial<T>, Partial<T>);

pub fn f<T>((b1, b2): (T, T), (b3, b4): (T, T), f: |T| -> T) -> Two<T> {
    let p = Partial { x: b1, y: b2 };
    let q = Partial { x: b3, y: b4 };

     // Move of `q` is legal even though we have already moved `q.y`;
     // the `..q` moves all fields *except* `q.y` in this context.
     // Likewise, the move of `p.x` is legal for similar reasons.
    (Partial { x: f(q.y), ..p }, Partial { y: f(p.x), ..q })
}

pub fn main() {
    let two = f((S::new(1), S::new(3)),
                (S::new(5), S::new(7)),
                |S { val: z }| S::new(z+1));
    assert_eq!(two, (Partial { x: S::new(8), y: S::new(3) },
                     Partial { x: S::new(5), y: S::new(2) }));
}
