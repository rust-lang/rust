// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


trait DigitCollection: Sized {
    type Iter: Iterator<Item = u8>;
    fn digit_iter(self) -> Self::Iter;

    fn digit_sum(self) -> u32 {
        self.digit_iter()
            .map(|digit: u8| digit as u32)
            .fold(0, |sum, digit| sum + digit)
    }
}

impl<I> DigitCollection for I where I: Iterator<Item=u8> {
    type Iter = I;

    fn digit_iter(self) -> I {
        self
    }
}

fn main() {
    let xs = vec![1, 2, 3, 4, 5];
    assert_eq!(xs.into_iter().digit_sum(), 15);
}
