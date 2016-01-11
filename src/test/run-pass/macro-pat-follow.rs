// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

macro_rules! pat_in {
    ($p:pat in $e:expr) => {{
        let mut iter = $e.into_iter();
        while let $p = iter.next() {}
    }}
}

macro_rules! pat_if {
    ($p:pat if $e:expr) => {{
        match Some(1u8) {
            $p if $e => {},
            _ => {}
        }
    }}
}

macro_rules! pat_bar {
    ($p:pat | $p2:pat) => {{
        match Some(1u8) {
            $p | $p2 => {},
            _ => {}
        }
    }}
}

fn main() {
    pat_in!(Some(_) in 0..10);
    pat_if!(Some(x) if x > 0);
    pat_bar!(Some(1u8) | None);
}
