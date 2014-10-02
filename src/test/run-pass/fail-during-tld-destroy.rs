// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::task;

static mut DROPS: uint = 0;

struct Foo;
impl Drop for Foo {
    fn drop(&mut self) {
        unsafe { DROPS += 1; }
        fail!()
    }
}

fn main() {
    let _ = task::try(proc() {
        local_data_key!(foo: Foo);
        foo.replace(Some(Foo));
    });

    unsafe {
        assert_eq!(DROPS, 1);
    }
}

