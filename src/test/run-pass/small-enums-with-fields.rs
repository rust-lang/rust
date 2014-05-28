// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(macro_rules)]

extern crate debug;

use std::mem::size_of;

#[deriving(Eq, Show)]
enum Either<T, U> { Left(T), Right(U) }

macro_rules! check {
    ($t:ty, $sz:expr, $($e:expr, $s:expr),*) => {{
        assert_eq!(size_of::<$t>(), $sz);
        $({
            static S: $t = $e;
            let v: $t = $e;
            assert_eq!(S, v);
            assert_eq!(format!("{:?}", v).as_slice(), $s);
            assert_eq!(format!("{:?}", S).as_slice(), $s);
        });*
    }}
}

pub fn main() {
    check!(Option<u8>, 2,
           None, "None",
           Some(129u8), "Some(129u8)");
    check!(Option<i16>, 4,
           None, "None",
           Some(-20000i16), "Some(-20000i16)");
    check!(Either<u8, i8>, 2,
           Left(132u8), "Left(132u8)",
           Right(-32i8), "Right(-32i8)");
    check!(Either<u8, i16>, 4,
           Left(132u8), "Left(132u8)",
           Right(-20000i16), "Right(-20000i16)");
}
