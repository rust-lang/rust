// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// run-pass
// Check that the object bound dyn A + 'a: A is preferred over the
// where clause bound dyn A + 'static: A.

#![allow(unused)]

trait A {
    fn test(&self);
}

fn foo(x: &dyn A)
where
    dyn A + 'static: A, // Using this bound would lead to a lifetime error.
{
    x.test();
}

fn main () {}
