// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Ensure constant and array length values are not taken from source
// code, which wreaks havoc with macros.

#![feature(associated_consts)]

macro_rules! make {
    ($n:expr) => {
        pub struct S;

        // @has issue_33302/constant.CST.html \
        //        '//pre[@class="rust const"]' 'pub const CST: i32 = 4 * 4'
        pub const CST: i32 = ($n * $n);
        // @has issue_33302/static.ST.html \
        //        '//pre[@class="rust static"]' 'pub static ST: i32 = 4 * 4'
        pub static ST: i32 = ($n * $n);

        pub trait T<X> {
            fn ignore(_: &X) {}
            const C: X;
            // @has issue_33302/trait.T.html \
            //        '//*[@class="rust trait"]' 'const D: i32 = 4 * 4;'
            // @has - '//*[@id="associatedconstant.D"]' 'const D: i32 = 4 * 4'
            const D: i32 = ($n * $n);
        }

        // @has issue_33302/struct.S.html \
        //        '//h3[@class="impl"]' 'impl T<[i32; 16]> for S'
        // @has - '//*[@id="associatedconstant.C"]' 'const C: [i32; 16] = [0; 4 * 4]'
        // @has - '//*[@id="associatedconstant.D"]' 'const D: i32 = 4 * 4'
        impl T<[i32; ($n * $n)]> for S {
            const C: [i32; ($n * $n)] = [0; ($n * $n)];
        }
    }
}

make!(4);
