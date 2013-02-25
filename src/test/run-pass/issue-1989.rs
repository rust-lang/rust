// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// exec-env:RUST_CC_ZEAL=1

enum maybe_pointy {
    none,
    p(@mut Pointy)
}

struct Pointy {
    a : maybe_pointy,
    f : fn@()->(),
}

fn empty_pointy() -> @mut Pointy {
    return @mut Pointy{
        a : none,
        f : fn@()->(){},
    }
}

pub fn main() {
    let v = ~[empty_pointy(), empty_pointy()];
    v[0].a = p(v[0]);
}
