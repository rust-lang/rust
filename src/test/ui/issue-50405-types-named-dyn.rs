// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #50405: do not permit traits or other types named `dyn` to
// resolve parsing ambiguity.

mod a {
    trait dyn { //~ ERROR types cannot be named `dyn`
    }
}

mod b {
    struct dyn { } //~ ERROR types cannot be named `dyn`
}

mod c {
    enum dyn { } //~ ERROR types cannot be named `dyn`
}

mod d {
    union dyn { _f: () } //~ ERROR types cannot be named `dyn`
}

mod e {
    type dyn = u32; //~ ERROR types cannot be named `dyn`
}

fn main() {}
