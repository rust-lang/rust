// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn mangle(x: &mut Option<u32>) -> bool { *x = None; false }

fn main() {
    let ref mut x = Some(4);
    match x {
        &mut None => {}
        &mut Some(_) if
            mangle(
                x //~ ERROR E0301
                )
            => {}
        &mut Some(_) => {}
    }
}
