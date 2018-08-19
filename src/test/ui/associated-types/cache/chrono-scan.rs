// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_attrs)]
#![allow(warnings)]

pub type ParseResult<T> = Result<T, ()>;

pub enum Item<'a> {     Literal(&'a str),
 }

pub fn colon_or_space(s: &str) -> ParseResult<&str> {
    unimplemented!()
}

pub fn timezone_offset_zulu<F>(s: &str, colon: F) -> ParseResult<(&str, i32)>
        where F: FnMut(&str) -> ParseResult<&str> {
    unimplemented!()
}

pub fn parse<'a, I>(mut s: &str, items: I) -> ParseResult<()>
        where I: Iterator<Item=Item<'a>> {
    macro_rules! try_consume {
        ($e:expr) => ({ let (s_, v) = try!($e); s = s_; v })
    }
    let offset = try_consume!(timezone_offset_zulu(s.trim_left(), colon_or_space));
    let offset = try_consume!(timezone_offset_zulu(s.trim_left(), colon_or_space));
    Ok(())
}

#[rustc_error]
fn main() { } //~ ERROR compilation successful
