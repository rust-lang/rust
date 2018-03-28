// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum Wrapper {
    Wrap(i32),
}

use Wrapper::Wrap;

pub fn main() {
    let Wrap(x) = &Wrap(3);
    println!("{}", *x);

    let Wrap(x) = &mut Wrap(3);
    println!("{}", *x);

    if let Some(x) = &Some(3) {
        println!("{}", *x);
    } else {
        panic!();
    }

    if let Some(x) = &mut Some(3) {
        println!("{}", *x);
    } else {
        panic!();
    }

    if let Some(x) = &mut Some(3) {
        *x += 1;
    } else {
        panic!();
    }

    while let Some(x) = &Some(3) {
        println!("{}", *x);
        break;
    }
    while let Some(x) = &mut Some(3) {
        println!("{}", *x);
        break;
    }
    while let Some(x) = &mut Some(3) {
        *x += 1;
        break;
    }
}
