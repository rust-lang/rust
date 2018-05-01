// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -g -C no-prepopulate-passes
// ignore-tidy-linelength
// min-llvm-version 4.0

#![crate_type = "lib"]

#[no_mangle]
pub fn foo() -> ! {
// CHECK: @foo() unnamed_addr #0
    loop {}
}

pub enum EmptyEnum {}

#[no_mangle]
pub fn bar() -> EmptyEnum {
// CHECK: @bar() unnamed_addr #0
    loop {}
}

// CHECK: attributes #0 = {{{.*}} noreturn {{.*}}}

// CHECK: DISubprogram(name: "foo", {{.*}} DIFlagNoReturn
// CHECK: DISubprogram(name: "bar", {{.*}} DIFlagNoReturn
