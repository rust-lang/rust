// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that values are not leaked when a dtor panics (#14875)

use std::panic::{self, UnwindSafe};

struct SetInnerOnDrop<'a>(&'a mut bool);

impl<'a> UnwindSafe for SetInnerOnDrop<'a> {}

impl<'a> Drop for SetInnerOnDrop<'a> {
    fn drop(&mut self) {
        *self.0 = true;
    }
}

struct PanicOnDrop;
impl Drop for PanicOnDrop {
    fn drop(&mut self) {
        panic!("test panic");
    }
}


fn main() {
    let mut set_on_drop = false;
    {
        let set_inner_on_drop = SetInnerOnDrop(&mut set_on_drop);
        let _ = panic::catch_unwind(|| {
            let _set_inner_on_drop = set_inner_on_drop;
            let _panic_on_drop = PanicOnDrop;
        });
    }
    assert!(set_on_drop);
}
