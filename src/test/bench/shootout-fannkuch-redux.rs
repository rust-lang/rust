// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::os;
use std::slice;

fn max(a: i32, b: i32) -> i32 {
    if a > b {
        a
    } else {
        b
    }
}

fn fannkuch_redux(n: i32) -> i32 {
    let mut perm = slice::from_elem(n as uint, 0i32);
    let mut perm1 = slice::from_fn(n as uint, |i| i as i32);
    let mut count = slice::from_elem(n as uint, 0i32);
    let mut max_flips_count = 0i32;
    let mut perm_count = 0i32;
    let mut checksum = 0i32;

    let mut r = n;
    loop {
        while r != 1 {
            count[r - 1] = r;
            r -= 1;
        }

        for (perm_i, perm1_i) in perm.mut_iter().zip(perm1.iter()) {
            *perm_i = *perm1_i;
        }

        let mut flips_count: i32 = 0;
        let mut k: i32;
        loop {
            k = perm[0];
            if k == 0 {
                break;
            }

            let k2 = (k+1) >> 1;
            for i in range(0i32, k2) {
                perm.swap(i as uint, (k - i) as uint);
            }
            flips_count += 1;
        }

        max_flips_count = max(max_flips_count, flips_count);
        checksum += if perm_count % 2 == 0 {
            flips_count
        } else {
            -flips_count
        };

        // Use incremental change to generate another permutation.
        loop {
            if r == n {
                println!("{}", checksum);
                return max_flips_count;
            }

            let perm0 = perm1[0];
            let mut i: i32 = 0;
            while i < r {
                let j = i + 1;
                perm1[i] = perm1[j];
                i = j;
            }
            perm1[r] = perm0;

            count[r] -= 1;
            if count[r] > 0 {
                break;
            }
            r += 1;
        }

        perm_count += 1;
    }
}

fn main() {
    let args = os::args();
    let n = if args.len() > 1 {
        from_str::<i32>(args[1]).unwrap()
    } else {
        2
    };
    println!("Pfannkuchen({}) = {}", n as int, fannkuch_redux(n) as int);
}
