// The Computer Language Benchmarks Game
// http://benchmarksgame.alioth.debian.org/
//
// contributed by the Rust Project Developers

// Copyright (c) 2013-2014 The Rust Project Developers
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

// ignore-pretty very bad with line comments
// ignore-android doesn't terminate?

#![feature(slicing_syntax)]

use std::iter::range_step;
use std::io::{stdin, stdout, File};

static LINE_LEN: uint = 60;

fn make_complements() -> [u8, ..256] {
    let transforms = [
        ('A', 'T'), ('C', 'G'), ('G', 'C'), ('T', 'A'),
        ('U', 'A'), ('M', 'K'), ('R', 'Y'), ('W', 'W'),
        ('S', 'S'), ('Y', 'R'), ('K', 'M'), ('V', 'B'),
        ('H', 'D'), ('D', 'H'), ('B', 'V'), ('N', 'N'),
        ('\n', '\n')];
    let mut complements: [u8, ..256] = [0, ..256];
    for (i, c) in complements.iter_mut().enumerate() {
        *c = i as u8;
    }
    let lower = 'A' as u8 - 'a' as u8;
    for &(from, to) in transforms.iter() {
        complements[from as uint] = to as u8;
        complements[(from as u8 - lower) as uint] = to as u8;
    }
    complements
}

fn main() {
    let complements = make_complements();
    let data = if std::os::getenv("RUST_BENCH").is_some() {
        File::open(&Path::new("shootout-k-nucleotide.data")).read_to_end()
    } else {
        stdin().read_to_end()
    };
    let mut data = data.unwrap();

    for seq in data.as_mut_slice().split_mut(|c| *c == '>' as u8) {
        // skip header and last \n
        let begin = match seq.iter().position(|c| *c == '\n' as u8) {
            None => continue,
            Some(c) => c
        };
        let len = seq.len();
        let seq = seq[mut begin+1..len-1];

        // arrange line breaks
        let len = seq.len();
        let off = LINE_LEN - len % (LINE_LEN + 1);
        for i in range_step(LINE_LEN, len, LINE_LEN + 1) {
            for j in std::iter::count(i, -1).take(off) {
                seq[j] = seq[j - 1];
            }
            seq[i - off] = '\n' as u8;
        }

        // reverse complement, as
        //    seq.reverse(); for c in seq.iter_mut() { *c = complements[*c] }
        // but faster:
        for (front, back) in two_side_iter(seq) {
            let tmp = complements[*front as uint];
            *front = complements[*back as uint];
            *back = tmp;
        }
        if seq.len() % 2 == 1 {
            let middle = &mut seq[seq.len() / 2];
            *middle = complements[*middle as uint];
        }
    }

    stdout().write(data.as_slice()).unwrap();
}

pub struct TwoSideIter<'a, T: 'a> {
    first: *mut T,
    last: *mut T,
    marker: std::kinds::marker::ContravariantLifetime<'a>,
    marker2: std::kinds::marker::NoCopy
}

pub fn two_side_iter<'a, T>(slice: &'a mut [T]) -> TwoSideIter<'a, T> {
    let len = slice.len();
    let first = slice.as_mut_ptr();
    let last = if len == 0 {
        first
    } else {
        unsafe { first.offset(len as int - 1) }
    };

    TwoSideIter {
        first: first,
        last: last,
        marker: std::kinds::marker::ContravariantLifetime,
        marker2: std::kinds::marker::NoCopy
    }
}

impl<'a, T> Iterator<(&'a mut T, &'a mut T)> for TwoSideIter<'a, T> {
    fn next(&mut self) -> Option<(&'a mut T, &'a mut T)> {
        if self.first < self.last {
            let result = unsafe { (&mut *self.first, &mut *self.last) };
            self.first = unsafe { self.first.offset(1) };
            self.last = unsafe { self.last.offset(-1) };
            Some(result)
        } else {
            None
        }
    }
}
