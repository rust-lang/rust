// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait x {
    fn use_x<T>(&self);
}
struct y(());
impl x for y {
    fn use_x<T>(&self) {
        struct foo { //~ ERROR quux
            i: ()
        }
        fn new_foo<T>(i: ()) -> foo {
            foo { i: i }
        }
    }
}
