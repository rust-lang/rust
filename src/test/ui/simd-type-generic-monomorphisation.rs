// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(repr_simd, platform_intrinsics)]

// error-pattern:monomorphising SIMD type `Simd2<X>` with a non-machine element type `X`

struct X(Vec<i32>);
#[repr(simd)]
struct Simd2<T>(T, T);

fn main() {
    let _ = Simd2(X(vec![]), X(vec![]));
}
