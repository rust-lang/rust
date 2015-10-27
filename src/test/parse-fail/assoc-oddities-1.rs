// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z parse-only

fn that_odd_parse() {
    // following lines below parse and must not fail
    x = if c { a } else { b }();
    x <- if c { a } else { b }[n];
    x = if true { 1 } else { 0 } as *mut _;
    // however this does not parse and probably should fail to retain compat?
    // NB: `..` here is arbitrary, failure happens/should happen ∀ops that aren’t `=` or `<-`
    // see assoc-oddities-2 and assoc-oddities-3
    ..if c { a } else { b }[n]; //~ ERROR expected one of
}
