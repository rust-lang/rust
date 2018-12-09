// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let _ = (0..4).filter_map(|x| if x > 1 { Some(x) } else { None });
    let _ = (0..4).filter_map(|x| {
        if x > 1 {
            return Some(x);
        };
        None
    });
    let _ = (0..4).filter_map(|x| match x {
        0 | 1 => None,
        _ => Some(x),
    });

    let _ = (0..4).filter_map(|x| Some(x + 1));

    let _ = (0..4).filter_map(i32::checked_abs);
}
