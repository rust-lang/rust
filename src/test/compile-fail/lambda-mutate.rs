// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern:assigning to captured outer variable in a heap closure
// Make sure we can't write to upvars from fn@s
fn main() {
    let i = 0;
    let ctr = fn@ () -> int { i = i + 1; return i; };
    log(error, ctr());
    log(error, ctr());
    log(error, ctr());
    log(error, ctr());
    log(error, ctr());
    log(error, i);
}
