// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This is a regression test for a problem encountered around upvar
// inference and trait caching: in particular, we were entering a
// temporary closure kind during inference, and then caching results
// based on that temporary kind, which led to no error being reported
// in this particular test.

fn main() {
    let inc = || {};
    inc();

    fn apply<F>(f: F) where F: Fn() {
        f()
    }

    let mut farewell = "goodbye".to_owned();
    let diary = || { //~ ERROR E0525
        farewell.push_str("!!!");
        println!("Then I screamed {}.", farewell);
    };

    apply(diary);
}
