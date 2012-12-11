// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that pure functions cannot modify aliased state.

pure fn modify_in_ref(&&sum: {mut f: int}) {
    sum.f = 3; //~ ERROR assigning to mutable field prohibited in pure context
}

pure fn modify_in_box(sum: @mut {f: int}) {
    sum.f = 3; //~ ERROR assigning to mutable field prohibited in pure context
}

trait modify_in_box_rec {
    pure fn modify_in_box_rec(sum: @{mut f: int});
}

impl int: modify_in_box_rec {
    pure fn modify_in_box_rec(sum: @{mut f: int}) {
        sum.f = self; //~ ERROR assigning to mutable field prohibited in pure context
    }
}

fn main() {
}
