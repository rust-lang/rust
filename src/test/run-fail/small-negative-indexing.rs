// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern:index out of bounds: the len is 1024 but the index is -1

use std::vec;

fn main() {
    let v = vec::from_fn(1024u, {|n| n});
    // this should trip a bounds check
    error2!("{:?}", v[-1i8]);
}
