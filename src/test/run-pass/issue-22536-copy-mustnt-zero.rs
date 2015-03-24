// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for Issue #22536: If a type implements Copy, then
// moving it must not zero the original memory.

// pretty-expanded FIXME #23616

trait Resources {
    type Buffer: Copy;
    fn foo(&self) {}
}

struct BufferHandle<R: Resources> {
    raw: <R as Resources>::Buffer,
}
impl<R: Resources> Copy for BufferHandle<R> {}

enum Res {}
impl Resources for Res {
    type Buffer = u32;
}
impl Copy for Res { }

fn main() {
    let b: BufferHandle<Res> = BufferHandle { raw: 1 };
    let c = b;
    assert_eq!(c.raw, b.raw)
}
