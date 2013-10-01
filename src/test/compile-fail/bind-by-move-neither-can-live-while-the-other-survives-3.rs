// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct X { x: (), }

impl Drop for X {
    fn drop(&mut self) {
        error2!("destructor runs");
    }
}

enum double_option<T,U> { some2(T,U), none2 }

fn main() {
    let x = some2(X { x: () }, X { x: () });
    match x {
        some2(ref _y, _z) => { }, //~ ERROR cannot bind by-move and by-ref in the same pattern
        none2 => fail2!()
    }
}
