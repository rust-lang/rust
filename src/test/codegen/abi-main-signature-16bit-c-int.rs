// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Checks the signature of the implicitly generated native main()
// entry point. It must match C's `int main(int, char **)`.

// This test is for targets with 16bit c_int only.
// ignore-aarch64
// ignore-arm
// ignore-asmjs
// ignore-hexagon
// ignore-mips
// ignore-mips64
// ignore-powerpc
// ignore-powerpc64
// ignore-s390x
// ignore-sparc
// ignore-sparc64
// ignore-wasm32
// ignore-x86
// ignore-x86_64
// ignore-xcore

fn main() {
}

// CHECK: define i16 @main(i16, i8**)
