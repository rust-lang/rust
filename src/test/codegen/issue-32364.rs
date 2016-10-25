// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-arm
// ignore-aarch64

// compile-flags: -C no-prepopulate-passes

struct Foo;

impl Foo {
// CHECK: define internal x86_stdcallcc void @{{.*}}foo{{.*}}()
    #[inline(never)]
    pub extern "stdcall" fn foo<T>() {
    }
}

fn main() {
    Foo::foo::<Foo>();
}
