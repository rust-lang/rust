// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that removed LLVM SIMD intrinsics continue
// to work via the "AutoUpgrade" mechanism.

#![feature(cfg_target_feature, repr_simd)]
#![feature(platform_intrinsics, stmt_expr_attributes)]

#[repr(simd)]
#[derive(PartialEq, Debug)]
struct i16x8(i16, i16, i16, i16, i16, i16, i16, i16);

fn main() {
    #[cfg(target_feature = "sse2")] unsafe {
        extern "platform-intrinsic" {
            fn x86_mm_min_epi16(x: i16x8, y: i16x8) -> i16x8;
        }
        assert_eq!(x86_mm_min_epi16(i16x8(0, 1, 2, 3, 4, 5, 6, 7),
                                    i16x8(7, 6, 5, 4, 3, 2, 1, 0)),
                                    i16x8(0, 1, 2, 3, 3, 2, 1, 0));
    };
}
