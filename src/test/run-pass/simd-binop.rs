// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(phase, simd)];
#[allow(experimental)];

#[phase(syntax)] extern crate simd_syntax;
extern crate simd;
extern crate test;

use simd::{Simd, f64x4, boolx8};

#[inline(never)] fn test_int(e: i32) -> i32 {
    let v = gather_simd!(e, 0i32, 0i32, 0i32);
    (v * v + v - v)[0]
}

#[inline(never)] fn test_float(e: f32) -> f32 {
    let v = gather_simd!(e, 0f32, 0f32, 0f32);
    (v * v + v - v)[0]
}

#[inline(never)] pub fn test_shift(e: u32) -> u32 {
    let v = gather_simd!(e, 0u32, 0u32, 0u32);
    let one = gather_simd!(1u32, 0u32, 0u32, 0u32);
    (v << one >> one)[0]
}

#[inline(never)] fn fake_mutate<T>(_: &mut T) {}

#[inline(never)] fn f64x4_actions(lhs: f64x4, rhs: f64x4) -> (boolx8, boolx8, boolx8) {
    let llhs1 = swizzle_simd!(lhs > rhs .. lhs >= rhs -> (0, 1, 2, 3, 4, 5, 6, 7));
    let llhs2 = swizzle_simd!(lhs == rhs .. lhs != rhs -> (0, 1, 2, 3, 4, 5, 6, 7));
    let llhs3 = swizzle_simd!(lhs <= rhs .. lhs < rhs -> (0, 1, 2, 3, 4, 5, 6, 7));
    let m1 = swizzle_simd!(llhs1 .. llhs2 -> (0, 1, 2, 3, 4, 5,  6,  7,  8,  9, 10, 11));
    let m2 = swizzle_simd!(llhs2 .. llhs3 -> (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
    (swizzle_simd!(m1 .. m2 -> (0,  1,   2,  3,  4,  5,  6,  7)),
     swizzle_simd!(m1 .. m2 -> (8,  9,  10, 11, 12, 13, 14, 15)),
     swizzle_simd!(m1 .. m2 -> (16, 17, 18, 19, 20, 21, 22, 23)))
}

pub fn main() {
    assert_eq!(test_int(3i32), 9i32);
    assert_eq!(test_float(3f32), 9f32);
    assert_eq!(test_shift(3u32), 3u32);


    let mut v1 = gather_simd!(3.1415926535f64, 6.2821853070f64, 0.25f64, 0.25f64);
    let mut v2 = swizzle_simd!(v1 -> (1, 0, 2, 2));
    fake_mutate(&mut v1);
    fake_mutate(&mut v2);

    let (cond0a, cond0b, cond0c) = f64x4_actions(v1, v2);
    let cond1 = swizzle_simd!(cond0a -> (0,  1,  2,  3));
    let cond2 = swizzle_simd!(cond0a -> (4,  5,  6,  7));
    let cond3 = swizzle_simd!(cond0b -> (0,  1,  2,  3));
    let cond4 = swizzle_simd!(cond0b -> (4,  5,  6,  7));
    let cond5 = swizzle_simd!(cond0c -> (0,  1,  2,  3));
    let cond6 = swizzle_simd!(cond0c -> (4,  5,  6,  7));

    fn format_fold_f(string: ~str, (idx, v): (uint, &bool)) -> ~str {
        string + format!(" {:}:{:}", idx, v)
    }

    assert!(cond1[0] == false);
    assert!(cond1[1] == true );
    assert!(cond1[2] == false);
    assert!(cond1[3] == false);
    assert!(cond2[0] == false);
    assert!(cond2[1] == true );
    assert!(cond2[2] == true );
    assert!(cond2[3] == true );
    assert!(cond3[0] == false);
    assert!(cond3[1] == false);
    assert!(cond3[2] == true );
    assert!(cond3[3] == true );
    assert!(cond4[0] == true );
    assert!(cond4[1] == true );
    assert!(cond4[2] == false);
    assert!(cond4[3] == false);
    assert!(cond5[0] == true );
    assert!(cond5[1] == false);
    assert!(cond5[2] == true );
    assert!(cond5[3] == true );
    assert!(cond6[0] == true );
    assert!(cond6[1] == false);
    assert!(cond6[2] == false);
    assert!(cond6[3] == false);

    // watch out, these are long.
    let mut v1 = gather_simd!(3i8, 1i8, 4i8, 1i8, 5i8, 9i8, 2i8, 0i8,
                              6i8, 2i8, 8i8, 2i8, 1i8, 8i8, 5i8, 0i8);
    let mut v2 = gather_simd!(6i8, 2i8, 8i8, 2i8, 1i8, 8i8, 5i8, 0i8,
                              3i8, 1i8, 4i8, 1i8, 5i8, 9i8, 2i8, 0i8);
    fake_mutate(&mut v1);
    fake_mutate(&mut v2);

    println!("{}", v1.iter().fold(~"v1:", |string, v| string + format!(" {}", v) ));
    println!("{}", v2.iter().fold(~"v2:", |string, v| string + format!(" {}", v) ));

    let cond1 = v1 >  v2; assert_eq!(cond1.len(), 16);
    let cond2 = v1 >= v2; assert_eq!(cond2.len(), 16);
    let cond3 = v1 == v2; assert_eq!(cond3.len(), 16);
    let cond4 = v1 != v2; assert_eq!(cond4.len(), 16);
    let cond5 = v1 <= v2; assert_eq!(cond5.len(), 16);
    let cond6 = v1 <  v2; assert_eq!(cond6.len(), 16);

    println!("{}", cond1.iter().enumerate().fold(~"cond1:", format_fold_f));
    println!("{}", cond2.iter().enumerate().fold(~"cond2:", format_fold_f));
    println!("{}", cond3.iter().enumerate().fold(~"cond3:", format_fold_f));
    println!("{}", cond4.iter().enumerate().fold(~"cond4:", format_fold_f));
    println!("{}", cond5.iter().enumerate().fold(~"cond5:", format_fold_f));
    println!("{}", cond6.iter().enumerate().fold(~"cond6:", format_fold_f));

    assert!(cond1[0]  == false &&
            cond1[1]  == false &&
            cond1[2]  == false &&
            cond1[3]  == false &&
            cond1[4]  == true  &&
            cond1[5]  == true  &&
            cond1[6]  == false &&
            cond1[7]  == false &&
            cond1[8]  == true  &&
            cond1[9]  == true  &&
            cond1[10] == true  &&
            cond1[11] == true  &&
            cond1[12] == false &&
            cond1[13] == false &&
            cond1[14] == true  &&
            cond1[15] == false);
    assert!(cond2[0]  == false &&
            cond2[1]  == false &&
            cond2[2]  == false &&
            cond2[3]  == false &&
            cond2[4]  == true  &&
            cond2[5]  == true  &&
            cond2[6]  == false &&
            cond2[7]  == true  &&
            cond2[8]  == true  &&
            cond2[9]  == true  &&
            cond2[10] == true  &&
            cond2[11] == true  &&
            cond2[12] == false &&
            cond2[13] == false &&
            cond2[14] == true  &&
            cond2[15] == true);
    assert!(cond3[0]  == false &&
            cond3[1]  == false &&
            cond3[2]  == false &&
            cond3[3]  == false &&
            cond3[4]  == false &&
            cond3[5]  == false &&
            cond3[6]  == false &&
            cond3[7]  == true  &&
            cond3[8]  == false &&
            cond3[9]  == false &&
            cond3[10] == false &&
            cond3[11] == false &&
            cond3[12] == false &&
            cond3[13] == false &&
            cond3[14] == false &&
            cond3[15] == true);
    assert!(cond4[0]  == true  &&
            cond4[1]  == true  &&
            cond4[2]  == true  &&
            cond4[3]  == true  &&
            cond4[4]  == true  &&
            cond4[5]  == true  &&
            cond4[6]  == true  &&
            cond4[7]  == false &&
            cond4[8]  == true  &&
            cond4[9]  == true  &&
            cond4[10] == true  &&
            cond4[11] == true  &&
            cond4[12] == true  &&
            cond4[13] == true  &&
            cond4[14] == true  &&
            cond4[15] == false);
    assert!(cond5[0]  == true  &&
            cond5[1]  == true  &&
            cond5[2]  == true  &&
            cond5[3]  == true  &&
            cond5[4]  == false &&
            cond5[5]  == false &&
            cond5[6]  == true  &&
            cond5[7]  == true  &&
            cond5[8]  == false &&
            cond5[9]  == false &&
            cond5[10] == false &&
            cond5[11] == false &&
            cond5[12] == true  &&
            cond5[13] == true  &&
            cond5[14] == false &&
            cond5[15] == true);
    assert!(cond6[0]  == true  &&
            cond6[1]  == true  &&
            cond6[2]  == true  &&
            cond6[3]  == true  &&
            cond6[4]  == false &&
            cond6[5]  == false &&
            cond6[6]  == true  &&
            cond6[7]  == false &&
            cond6[8]  == false &&
            cond6[9]  == false &&
            cond6[10] == false &&
            cond6[11] == false &&
            cond6[12] == true  &&
            cond6[13] == true  &&
            cond6[14] == false &&
            cond6[15] == false);
}
