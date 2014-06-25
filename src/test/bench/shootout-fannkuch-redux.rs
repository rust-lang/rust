// The Computer Language Benchmarks Game
// http://benchmarksgame.alioth.debian.org/
//
// contributed by the Rust Project Developers

// Copyright (c) 2014 The Rust Project Developers
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// - Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
//
// - Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in
//   the documentation and/or other materials provided with the
//   distribution.
//
// - Neither the name of "The Computer Language Benchmarks Game" nor
//   the name of "The Computer Language Shootout Benchmarks" nor the
//   names of its contributors may be used to endorse or promote
//   products derived from this software without specific prior
//   written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
// OF THE POSSIBILITY OF SUCH DAMAGE.

use std::cmp::max;

fn fact(n: uint) -> uint {
    range(1, n + 1).fold(1, |accu, i| accu * i)
}

fn fannkuch(n: uint, i: uint) -> (int, int) {
    let mut perm = Vec::from_fn(n, |e| ((n + e - i) % n + 1) as i32);
    let mut tperm = perm.clone();
    let mut count = Vec::from_elem(n, 0u);
    let mut perm_count = 0i;
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
    let n = std::os::args().as_slice()
                           .get(1)
                           .and_then(|arg| from_str(arg.as_slice()))
                           .unwrap_or(2u);

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
