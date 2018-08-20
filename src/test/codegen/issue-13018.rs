// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -O

// A drop([...].clone()) sequence on an Rc should be a no-op
// In particular, no call to __rust_dealloc should be emitted
#![crate_type = "lib"]
use std::rc::Rc;

pub fn foo(t: &Rc<Vec<usize>>) {
// CHECK-NOT: __rust_dealloc
    drop(t.clone());
}
