// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.



// -*- rust -*-
use core::sys;

enum t { make_t(@int), clam, }

fn foo(s: @int) {
    let count = sys::refcount(s);
    let x: t = make_t(s); // ref up

    match x {
      make_t(y) => {
        log(debug, y); // ref up then down

      }
      _ => { debug!("?"); fail; }
    }
    log(debug, sys::refcount(s));
    assert (sys::refcount(s) == count + 1u);
    let _ = sys::refcount(s); // don't get bitten by last-use.
}

fn main() {
    let s: @int = @0; // ref up

    let count = sys::refcount(s);

    foo(s); // ref up then down

    log(debug, sys::refcount(s));
    let count2 = sys::refcount(s);
    let _ = sys::refcount(s); // don't get bitten by last-use.
    assert count == count2;
}
