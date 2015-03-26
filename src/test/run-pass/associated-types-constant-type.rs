// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #23616

trait SignedUnsigned {
    type Opposite;
    fn convert(self) -> Self::Opposite;
}

impl SignedUnsigned for isize {
    type Opposite = usize;

    fn convert(self) -> usize {
        self as usize
    }
}

impl SignedUnsigned for usize {
    type Opposite = isize;

    fn convert(self) -> isize {
        self as isize
    }
}

fn get(x: isize) -> <isize as SignedUnsigned>::Opposite {
    x.convert()
}

fn main() {
    let x = get(22);
    assert_eq!(22, x);
}
