// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub trait Stream {
    type Item;
    type Error;
}

pub trait ParseError<I> {
    type Output;
}

impl ParseError<char> for u32 {
    type Output = ();
}

impl Stream for () {
    type Item = char;
    type Error = u32;
}

pub struct Lex<'a, I>
    where I: Stream,
          I::Error: ParseError<char>,
          <<I as Stream>::Error as ParseError<char>>::Output: 'a
{
    x: &'a <I::Error as ParseError<char>>::Output
}

pub struct Reserved<'a, I> where
    I: Stream<Item=char> + 'a,
    I::Error: ParseError<I::Item>,
    <<I as Stream>::Error as ParseError<char>>::Output: 'a

{
    x: Lex<'a, I>
}

fn main() {
    let r: Reserved<()> = Reserved {
        x: Lex {
            x: &()
        }
    };

    let _v = r.x.x;
}
