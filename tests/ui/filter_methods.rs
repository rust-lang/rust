// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::missing_docs_in_private_items)]

fn main() {
    let _: Vec<_> = vec![5; 6].into_iter().filter(|&x| x == 0).map(|x| x * 2).collect();

    let _: Vec<_> = vec![5_i8; 6]
        .into_iter()
        .filter(|&x| x == 0)
        .flat_map(|x| x.checked_mul(2))
        .collect();

    let _: Vec<_> = vec![5_i8; 6]
        .into_iter()
        .filter_map(|x| x.checked_mul(2))
        .flat_map(|x| x.checked_mul(2))
        .collect();

    let _: Vec<_> = vec![5_i8; 6]
        .into_iter()
        .filter_map(|x| x.checked_mul(2))
        .map(|x| x.checked_mul(2))
        .collect();
}
