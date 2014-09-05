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

use std::{cmp, iter, mem};
use std::sync::Future;

fn rotate(x: &mut [i32]) {
    let mut prev = x[0];
    for place in x.mut_iter().rev() {
        prev = mem::replace(place, prev)
    }
}

fn next_permutation(perm: &mut [i32], count: &mut [i32]) {
    for i in range(1, perm.len()) {
        rotate(perm.mut_slice_to(i + 1));
        let count_i = &mut count[i];
        if *count_i >= i as i32 {
            *count_i = 0;
        } else {
            *count_i += 1;
            break
        }
    }
}

struct P {
    p: [i32, .. 16],
}

struct Perm {
    cnt: [i32, .. 16],
    fact: [u32, .. 16],
    n: u32,
    permcount: u32,
    perm: P,
}

impl Perm {
    fn new(n: u32) -> Perm {
        let mut fact = [1, .. 16];
        for i in range(1, n as uint + 1) {
            fact[i] = fact[i - 1] * i as u32;
        }
        Perm {
            cnt: [0, .. 16],
            fact: fact,
            n: n,
            permcount: 0,
            perm: P { p: [0, .. 16 ] }
        }
    }

    fn get(&mut self, mut idx: i32) -> P {
        let mut pp = [0u8, .. 16];
        self.permcount = idx as u32;
        for (i, place) in self.perm.p.mut_iter().enumerate() {
            *place = i as i32 + 1;
        }

        for i in range(1, self.n as uint).rev() {
            let d = idx / self.fact[i] as i32;
            self.cnt[i] = d;
            idx %= self.fact[i] as i32;
            for (place, val) in pp.mut_iter().zip(self.perm.p.slice_to(i + 1).iter()) {
                *place = (*val) as u8
            }

            let d = d as uint;
            for j in range(0, i + 1) {
                self.perm.p[j] = if j + d <= i {pp[j + d]} else {pp[j+d-i-1]} as i32;
            }
        }

        self.perm
    }

    fn count(&self) -> u32 { self.permcount }
    fn max(&self) -> u32 { self.fact[self.n as uint] }

    fn next(&mut self) -> P {
        next_permutation(self.perm.p, self.cnt);
        self.permcount += 1;

        self.perm
    }
}


fn reverse(tperm: &mut [i32], mut k: uint) {
    tperm.mut_slice_to(k).reverse()
}

fn work(mut perm: Perm, n: uint, max: uint) -> (i32, i32) {
    let mut checksum = 0;
    let mut maxflips = 0;

    let mut p = perm.get(n as i32);

    while perm.count() < max as u32 {
        let mut flips = 0;

        while p.p[0] != 1 {
            let k = p.p[0] as uint;
            reverse(p.p, k);
            flips += 1;
        }

        checksum += if perm.count() % 2 == 0 {flips} else {-flips};
        maxflips = cmp::max(maxflips, flips);

        p = perm.next();
    }

    (checksum, maxflips)
}

fn fannkuch(n: i32) -> (i32, i32) {
    let perm = Perm::new(n as u32);

    let N = 4;
    let mut futures = vec![];
    let k = perm.max() / N;

    for (i, j) in range(0, N).zip(iter::count(0, k)) {
        let max = cmp::min(j+k, perm.max());

        futures.push(Future::spawn(proc() {
            work(perm, j as uint, max as uint)
        }))
    }

    let mut checksum = 0;
    let mut maxflips = 0;
    for fut in futures.mut_iter() {
        let (cs, mf) = fut.get();
        checksum += cs;
        maxflips = cmp::max(maxflips, mf);
    }
    (checksum, maxflips)
}

fn main() {
    let n = std::os::args().as_slice()
        .get(1)
        .and_then(|arg| from_str(arg.as_slice()))
        .unwrap_or(2i32);

    let (checksum, maxflips) = fannkuch(n);
    println!("{}\nPfannkuchen({}) = {}", checksum, n, maxflips);
}
