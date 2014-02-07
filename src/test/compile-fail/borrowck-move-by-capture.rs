// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub fn main() {
    // FIXME(#2202) - Due to the way that borrowck treats closures,
    // you get two error reports here.
    let bar = ~3;
    let _g = || { //~ ERROR capture of moved value
        let _h: proc() -> int = proc() *bar; //~ ERROR capture of moved value
    };
}
