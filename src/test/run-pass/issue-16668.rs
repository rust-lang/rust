// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unknown_features)]

struct Parser<'a, I, O> {
    parse: Box<FnMut(I) -> Result<O, String> + 'a>
}

impl<'a, I: 'a, O: 'a> Parser<'a, I, O> {
    fn compose<K: 'a>(mut self, mut rhs: Parser<'a, O, K>) -> Parser<'a, I, K> {
        // FIXME (#22405): Replace `Box::new` with `box` here when/if possible.
        Parser {
            parse: Box::new(move |x: I| {
                match (self.parse)(x) {
                    Ok(r) => (rhs.parse)(r),
                    Err(e) => Err(e)
                }
            })
        }
    }
}

fn main() {}
