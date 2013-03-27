// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum yes0<'lt> {
    // This will eventually be legal (and in fact the only way):
    X3(&'lt uint) //~ ERROR Illegal lifetime 'lt: only 'self is allowed
}

enum yes1<'self> {
    X4(&'self uint)
}

enum yes2 {
    X5(&'foo uint) //~ ERROR Illegal lifetime 'foo: only 'self is allowed
}

fn main() {}
