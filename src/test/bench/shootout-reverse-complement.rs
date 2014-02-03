// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-android doesn't terminate?

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
    for (i, c) in complements.mut_iter().enumerate() {
        *c = i as u8;
    }
    let lower = 'A' as u8 - 'a' as u8;
    for &(from, to) in transforms.iter() {
        complements[from as u8] = to as u8;
        complements[from as u8 - lower] = to as u8;
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

    for seq in data.mut_split(|c| *c == '>' as u8) {
        // skip header and last \n
        let begin = match seq.iter().position(|c| *c == '\n' as u8) {
            None => continue,
            Some(c) => c
        };
        let len = seq.len();
        let seq = seq.mut_slice(begin + 1, len - 1);

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
        //    seq.reverse(); for c in seq.mut_iter() {*c = complements[*c]}
        // but faster:
        let mut it = seq.mut_iter();
        loop {
            match (it.next(), it.next_back()) {
                (Some(front), Some(back)) => {
                    let tmp = complements[*front];
                    *front = complements[*back];
                    *back = tmp;
                }
                _ => break // vector exhausted.
            }
        }
    }

    stdout().write(data);
}
