// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum a_tag {
    a_tag(u64)
}

struct t_rec {
    c8: u8,
    t: a_tag
}

pub fn main() {
    let x = t_rec {c8: 22u8, t: a_tag(44u64)};
    let y = fmt!("%?", x);
    debug!("y = %s", y);
    assert_eq!(y, ~"{c8: 22, t: a_tag(44)}");
}
