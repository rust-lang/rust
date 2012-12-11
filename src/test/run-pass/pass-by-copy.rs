// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn magic(+x: {a: @int}) { log(debug, x); }
fn magic2(+x: @int) { log(debug, x); }

fn main() {
    let a = {a: @10}, b = @10;
    magic(a); magic({a: @20});
    magic2(b); magic2(@20);
}
