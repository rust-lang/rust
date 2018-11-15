// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use test::Bencher;

const CHARS: [char; 9] = ['0', 'x', '2', '5', 'A', 'f', '7', '8', '9'];
const RADIX: [u32; 5] = [2, 8, 10, 16, 32];

#[bench]
fn bench_to_digit_radix_2(b: &mut Bencher) {
    b.iter(|| CHARS.iter().cycle().take(10_000).map(|c| c.to_digit(2)).min())
}

#[bench]
fn bench_to_digit_radix_10(b: &mut Bencher) {
    b.iter(|| CHARS.iter().cycle().take(10_000).map(|c| c.to_digit(10)).min())
}

#[bench]
fn bench_to_digit_radix_16(b: &mut Bencher) {
    b.iter(|| CHARS.iter().cycle().take(10_000).map(|c| c.to_digit(16)).min())
}

#[bench]
fn bench_to_digit_radix_36(b: &mut Bencher) {
    b.iter(|| CHARS.iter().cycle().take(10_000).map(|c| c.to_digit(36)).min())
}

#[bench]
fn bench_to_digit_radix_var(b: &mut Bencher) {
    b.iter(|| CHARS.iter().cycle()
        .zip(RADIX.iter().cycle())
        .take(10_000)
        .map(|(c, radix)| c.to_digit(*radix)).min())
}
