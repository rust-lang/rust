// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// compile-flags: -Ccodegen-units=1 -O

fn linidx(row: usize, col: usize) -> usize {
    row * 1 + col * 3
}

fn main() {
    let mut mat = [1.0f32, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0];

    for i in 0..2 {
        for j in i+1..3 {
            if mat[linidx(j, 3)] > mat[linidx(i, 3)] {
                    for k in 0..4 {
                            let (x, rest) = mat.split_at_mut(linidx(i, k) + 1);
                            let a = x.last_mut().unwrap();
                            let b = rest.get_mut(linidx(j, k) - linidx(i, k) - 1).unwrap();
                            ::std::mem::swap(a, b);
                    }
            }
        }
    }
    assert_eq!([9.0, 5.0, 1.0, 10.0, 6.0, 2.0, 11.0, 7.0, 3.0, 12.0, 8.0, 4.0], mat);
}
