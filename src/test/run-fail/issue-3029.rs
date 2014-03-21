// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(unnecessary_allocation)];
#[allow(unreachable_code)];
#[allow(unused_variable)];


// error-pattern:so long
fn main() {
    let mut x = Vec::new();
    let y = vec!(3);
    fail!("so long");
    x.push_all_move(y);
    ~"good" + ~"bye";
}
