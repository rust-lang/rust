// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test makes sure that functions get annotated with the proper
// "target-cpu" attribute in LLVM.

// no-prefer-dynamic
// ignore-tidy-linelength
// compile-flags: -C no-prepopulate-passes -C panic=abort -Z cross-lang-lto -Cpasses=name-anon-globals

#![crate_type = "staticlib"]

// CHECK-LABEL: define {{.*}} @exported() {{.*}} #0
#[no_mangle]
pub extern fn exported() {
    not_exported();
}

// CHECK-LABEL: define {{.*}} @_ZN23target_cpu_on_functions12not_exported{{.*}}() {{.*}} #0
fn not_exported() {}

// CHECK: attributes #0 = {{.*}} "target-cpu"="{{.*}}"
