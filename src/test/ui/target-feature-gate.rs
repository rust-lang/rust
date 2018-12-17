// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
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
// ignore-wasm
// ignore-emscripten
// ignore-mips
// ignore-mips64
// gate-test-sse4a_target_feature
// gate-test-powerpc_target_feature
// gate-test-avx512_target_feature
// gate-test-tbm_target_feature
// gate-test-arm_target_feature
// gate-test-aarch64_target_feature
// gate-test-hexagon_target_feature
// gate-test-mips_target_feature
// gate-test-mmx_target_feature
// gate-test-wasm_target_feature
// gate-test-adx_target_feature
// gate-test-cmpxchg16b_target_feature
// min-llvm-version 6.0

#[target_feature(enable = "avx512bw")]
//~^ ERROR: currently unstable
unsafe fn foo() {
}

fn main() {}
