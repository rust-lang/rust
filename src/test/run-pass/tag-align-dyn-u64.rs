// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test

tag a_tag<A> {
    a_tag(A);
}

type t_rec = {
    c8: u8,
    t: a_tag<u64>
};

fn mk_rec() -> t_rec {
    return { c8:0u8, t:a_tag(0u64) };
}

fn is_8_byte_aligned(&&u: a_tag<u64>) -> bool {
    let p = ptr::addr_of(u) as uint;
    return (p & 7u) == 0u;
}

pub fn main() {
    let x = mk_rec();
    assert!(is_8_byte_aligned(x.t));
}
