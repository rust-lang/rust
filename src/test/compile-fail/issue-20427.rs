// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:i8.rs
extern crate i8;
use std::string as i16;
static i32: i32 = 0;
const i64: i64 = 0;
fn u8(f32: f32) {}
fn f<f64>(f64: f64) {}
//~^ ERROR user-defined types or type parameters cannot shadow the primitive types
type u16 = u16; //~ ERROR user-defined types or type parameters cannot shadow the primitive types
enum u32 {} //~ ERROR user-defined types or type parameters cannot shadow the primitive types
struct u64; //~ ERROR user-defined types or type parameters cannot shadow the primitive types
trait bool {} //~ ERROR user-defined types or type parameters cannot shadow the primitive types

mod char {
    extern crate i8;
    static i32_: i32 = 0;
    const i64_: i64 = 0;
    fn u8_(f32: f32) {}
    fn f_<f64_>(f64: f64_) {}
    type u16_ = u16;
    enum u32_ {}
    struct u64_;
    trait bool_ {}
    mod char_ {}

    mod str {
        use super::i8 as i8;
        use super::i32_ as i32;
        use super::i64_ as i64;
        use super::u8_ as u8;
        use super::f_ as f64;
        use super::u16_ as u16;
        //~^ ERROR user-defined types or type parameters cannot shadow the primitive types
        use super::u32_ as u32;
        //~^ ERROR user-defined types or type parameters cannot shadow the primitive types
        use super::u64_ as u64;
        //~^ ERROR user-defined types or type parameters cannot shadow the primitive types
        use super::bool_ as bool;
        //~^ ERROR user-defined types or type parameters cannot shadow the primitive types
        use super::{bool_ as str};
        //~^ ERROR user-defined types or type parameters cannot shadow the primitive types
        use super::char_ as char;
    }
}

trait isize_ {
    type isize; //~ ERROR user-defined types or type parameters cannot shadow the primitive types
}

fn usize<'usize>(usize: &'usize usize) -> &'usize usize { usize }

fn main() {
    let bool = true;
    match bool {
        str @ true => if str { i32 as i64 } else { i64 },
        false => i64,
    };
}
