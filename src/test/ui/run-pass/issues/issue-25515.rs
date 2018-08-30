// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::rc::Rc;

struct Foo<'r>(&'r mut i32);

impl<'r> Drop for Foo<'r> {
    fn drop(&mut self) {
        *self.0 += 1;
    }
}

fn main() {
    let mut drops = 0;

    {
        let _: Rc<Send> = Rc::new(Foo(&mut drops));
    }

    assert_eq!(1, drops);
}
