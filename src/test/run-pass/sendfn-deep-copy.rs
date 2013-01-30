// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() { test05(); }

fn mk_counter<A:Copy>() -> fn~(A) -> (A,uint) {
    // The only reason that the counter is generic is so that it closes
    // over both a type descriptor and some data.
    let mut v = ~[0u];
    return fn~(a: A) -> (A,uint) {
        let n = v[0];
        v[0] = n + 1u;
        (a, n)
    };
}

fn test05() {
    let fp0 = mk_counter::<float>();

    assert (5.3f, 0u) == fp0(5.3f);
    assert (5.5f, 1u) == fp0(5.5f);

    let fp1 = copy fp0;

    assert (5.3f, 2u) == fp0(5.3f);
    assert (5.3f, 2u) == fp1(5.3f);
    assert (5.5f, 3u) == fp0(5.5f);
    assert (5.5f, 3u) == fp1(5.5f);
}
