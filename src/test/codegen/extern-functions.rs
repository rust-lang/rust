// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(unwind_attributes)]

extern {
// CHECK: Function Attrs: nounwind
// CHECK-NEXT: declare void @extern_fn
    fn extern_fn();
// CHECK-NOT: Function Attrs: nounwind
// CHECK: declare void @unwinding_extern_fn
    #[unwind]
    fn unwinding_extern_fn();
}
