// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cmp::max;

fn fact(n: uint) -> uint {
    range(1, n + 1).fold(1, |accu, i| accu * i)
}

fn fannkuch(n: uint, i: uint) -> (int, int) {
    let mut perm = Vec::from_fn(n, |e| ((n + e - i) % n + 1) as i32);
    let mut tperm = perm.clone();
    let mut count = Vec::from_elem(n, 0u);
    let mut perm_count = 0;
    let mut checksum = 0;

    for countdown in range(1, fact(n - 1) + 1).rev() {
        for i in range(1, n) {
            let perm0 = *perm.get(0);
            for j in range(0, i) {
                *perm.get_mut(j) = *perm.get(j + 1);
            }
            *perm.get_mut(i) = perm0;

            let count_i = count.get_mut(i);
            if *count_i >= i {
                *count_i = 0;
            } else {
                *count_i += 1;
                break;
            }
        }

        tperm.clone_from(&perm);
        let mut flips_count = 0;
        loop {
            let k = *tperm.get(0);
            if k == 1 { break; }
            tperm.mut_slice_to(k as uint).reverse();
            flips_count += 1;
        }
        perm_count = max(perm_count, flips_count);
        checksum += if countdown & 1 == 1 {flips_count} else {-flips_count}
    }
    (checksum, perm_count)
}

fn main() {
    let n = std::os::args().get(1).and_then(|arg| from_str(*arg)).unwrap_or(2u);

    let (tx, rx) = channel();
    for i in range(0, n) {
        let tx = tx.clone();
        spawn(proc() tx.send(fannkuch(n, i)));
    }
    drop(tx);

    let mut checksum = 0;
    let mut perm = 0;
    for (cur_cks, cur_perm) in rx.iter() {
        checksum += cur_cks;
        perm = max(perm, cur_perm);
    }
    println!("{}\nPfannkuchen({}) = {}", checksum, n, perm);
}
