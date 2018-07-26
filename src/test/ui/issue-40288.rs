// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn save_ref<'a>(refr: &'a i32, to: &mut [&'a i32]) {
    for val in &mut *to {
        *val = refr;
    }
}

fn main() {
    let ref init = 0i32;
    let ref mut refr = 1i32;

    let mut out = [init];

    save_ref(&*refr, &mut out);

    // This shouldn't be allowed as `refr` is borrowed
    *refr = 3; //~ ERROR cannot assign to `*refr` because it is borrowed

    // Prints 3?!
    println!("{:?}", out[0]);
}
