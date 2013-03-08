// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #570

const lsl : int = 1 << 2;
const add : int = 1 + 2;
const addf : float = 1.0f + 2.0f;
const not : int = !0;
const notb : bool = !true;
const neg : int = -(1);

pub fn main() {
    fail_unless!((lsl == 4));
    fail_unless!((add == 3));
    fail_unless!((addf == 3.0f));
    fail_unless!((not == -1));
    fail_unless!((notb == false));
    fail_unless!((neg == -1));
}
