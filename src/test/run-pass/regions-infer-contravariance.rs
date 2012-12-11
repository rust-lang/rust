// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct boxed_int {
    f: &int,
}

fn get(bi: &r/boxed_int) -> &r/int {
    bi.f
}

fn with(bi: &r/boxed_int) {
    // Here, the upcast is allowed because the `boxed_int` type is
    // contravariant with respect to `&r`.  See also
    // compile-fail/regions-infer-invariance-due-to-mutability.rs
    let bi: &blk/boxed_int/&blk = bi;
    assert *get(bi) == 22;
}

fn main() {
    let g = 22;
    let foo = boxed_int { f: &g };
    with(&foo);
}