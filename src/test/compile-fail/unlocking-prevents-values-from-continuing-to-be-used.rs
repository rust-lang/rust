// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern mod extra;
use extra::sync::shared_mut::rwarc::RWArc;
use extra::sync::unlock::Unlock;

fn main() {
    let mut arc = RWArc::new(0u);
    let arc2 = arc.clone();
    let mut write_locked = arc.write_locked();
    let _value = write_locked.get();

    do write_locked.unlock { //~ ERROR cannot borrow `write_locked` as mutable more than once at a time
        let mut arc3 = arc2;

        // Can't bring a the other lock handle inside
        let mut write_locked = arc3.write_locked();
        let value = write_locked.get();

        // This breaks the sole ownership assumption given by
        // borrowing, and so must be banned because value is still in
        // scope.
        *value = 2;
    }
}
