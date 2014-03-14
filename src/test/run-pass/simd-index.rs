// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// checks that strict simd types are implicitly indexable.

#[feature(simd, phase)];
#[allow(experimental)];

#[phase(syntax)] extern crate simd_syntax;

def_type_simd!(#[allow(non_camel_case_types)] type f32x4 = <f32, ..4>;)
def_type_simd!(#[allow(non_camel_case_types)] type i32x4 = <i32, ..4>;)

#[inline(never)]
fn f32x4_index(v: &mut f32x4) {
    v[0] = 10.0f32;
}
#[inline(never)]
fn i32x4_index(v: &mut i32x4) {
    v[0] = 10i32;
}


pub fn main() {
    let mut v = gather_simd!(20.0f32, 20.0f32, 20.0f32, 20.0f32);
    f32x4_index(&mut v);
    //assert!(v[0] == 10.0f32);
    //assert!(v[1] == 20.0f32);

    let mut v = gather_simd!(15, 15, 15, 15);
    i32x4_index(&mut v);
    //assert!(v[0] == 10);
    //assert!(v[1] == 15);
}
