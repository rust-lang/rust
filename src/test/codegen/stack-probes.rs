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
// ignore-mips
// ignore-mips64
// ignore-powerpc
// ignore-powerpc64
// ignore-powerpc64le
// ignore-s390x
// ignore-sparc
// ignore-sparc64
// ignore-wasm
// ignore-emscripten
// ignore-windows
// min-system-llvm-version 5.0
// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]

#[no_mangle]
pub fn foo() {
// CHECK: @foo() unnamed_addr #0
// CHECK: attributes #0 = { {{.*}}"probe-stack"="__rust_probestack"{{.*}} }
}
