// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::libc;

pub fn main() {
  let f = 1 as *libc::FILE;
  info2!("{}", f as int);
  info2!("{}", f as uint);
  info2!("{}", f as i8);
  info2!("{}", f as i16);
  info2!("{}", f as i32);
  info2!("{}", f as i64);
  info2!("{}", f as u8);
  info2!("{}", f as u16);
  info2!("{}", f as u32);
  info2!("{}", f as u64);

  info2!("{}", 1 as int);
  info2!("{}", 1 as uint);
  info2!("{}", 1 as float);
  info2!("{}", 1 as *libc::FILE);
  info2!("{}", 1 as i8);
  info2!("{}", 1 as i16);
  info2!("{}", 1 as i32);
  info2!("{}", 1 as i64);
  info2!("{}", 1 as u8);
  info2!("{}", 1 as u16);
  info2!("{}", 1 as u32);
  info2!("{}", 1 as u64);
  info2!("{}", 1 as f32);
  info2!("{}", 1 as f64);

  info2!("{}", 1u as int);
  info2!("{}", 1u as uint);
  info2!("{}", 1u as float);
  info2!("{}", 1u as *libc::FILE);
  info2!("{}", 1u as i8);
  info2!("{}", 1u as i16);
  info2!("{}", 1u as i32);
  info2!("{}", 1u as i64);
  info2!("{}", 1u as u8);
  info2!("{}", 1u as u16);
  info2!("{}", 1u as u32);
  info2!("{}", 1u as u64);
  info2!("{}", 1u as f32);
  info2!("{}", 1u as f64);

  info2!("{}", 1i8 as int);
  info2!("{}", 1i8 as uint);
  info2!("{}", 1i8 as float);
  info2!("{}", 1i8 as *libc::FILE);
  info2!("{}", 1i8 as i8);
  info2!("{}", 1i8 as i16);
  info2!("{}", 1i8 as i32);
  info2!("{}", 1i8 as i64);
  info2!("{}", 1i8 as u8);
  info2!("{}", 1i8 as u16);
  info2!("{}", 1i8 as u32);
  info2!("{}", 1i8 as u64);
  info2!("{}", 1i8 as f32);
  info2!("{}", 1i8 as f64);

  info2!("{}", 1u8 as int);
  info2!("{}", 1u8 as uint);
  info2!("{}", 1u8 as float);
  info2!("{}", 1u8 as *libc::FILE);
  info2!("{}", 1u8 as i8);
  info2!("{}", 1u8 as i16);
  info2!("{}", 1u8 as i32);
  info2!("{}", 1u8 as i64);
  info2!("{}", 1u8 as u8);
  info2!("{}", 1u8 as u16);
  info2!("{}", 1u8 as u32);
  info2!("{}", 1u8 as u64);
  info2!("{}", 1u8 as f32);
  info2!("{}", 1u8 as f64);

  info2!("{}", 1i16 as int);
  info2!("{}", 1i16 as uint);
  info2!("{}", 1i16 as float);
  info2!("{}", 1i16 as *libc::FILE);
  info2!("{}", 1i16 as i8);
  info2!("{}", 1i16 as i16);
  info2!("{}", 1i16 as i32);
  info2!("{}", 1i16 as i64);
  info2!("{}", 1i16 as u8);
  info2!("{}", 1i16 as u16);
  info2!("{}", 1i16 as u32);
  info2!("{}", 1i16 as u64);
  info2!("{}", 1i16 as f32);
  info2!("{}", 1i16 as f64);

  info2!("{}", 1u16 as int);
  info2!("{}", 1u16 as uint);
  info2!("{}", 1u16 as float);
  info2!("{}", 1u16 as *libc::FILE);
  info2!("{}", 1u16 as i8);
  info2!("{}", 1u16 as i16);
  info2!("{}", 1u16 as i32);
  info2!("{}", 1u16 as i64);
  info2!("{}", 1u16 as u8);
  info2!("{}", 1u16 as u16);
  info2!("{}", 1u16 as u32);
  info2!("{}", 1u16 as u64);
  info2!("{}", 1u16 as f32);
  info2!("{}", 1u16 as f64);

  info2!("{}", 1i32 as int);
  info2!("{}", 1i32 as uint);
  info2!("{}", 1i32 as float);
  info2!("{}", 1i32 as *libc::FILE);
  info2!("{}", 1i32 as i8);
  info2!("{}", 1i32 as i16);
  info2!("{}", 1i32 as i32);
  info2!("{}", 1i32 as i64);
  info2!("{}", 1i32 as u8);
  info2!("{}", 1i32 as u16);
  info2!("{}", 1i32 as u32);
  info2!("{}", 1i32 as u64);
  info2!("{}", 1i32 as f32);
  info2!("{}", 1i32 as f64);

  info2!("{}", 1u32 as int);
  info2!("{}", 1u32 as uint);
  info2!("{}", 1u32 as float);
  info2!("{}", 1u32 as *libc::FILE);
  info2!("{}", 1u32 as i8);
  info2!("{}", 1u32 as i16);
  info2!("{}", 1u32 as i32);
  info2!("{}", 1u32 as i64);
  info2!("{}", 1u32 as u8);
  info2!("{}", 1u32 as u16);
  info2!("{}", 1u32 as u32);
  info2!("{}", 1u32 as u64);
  info2!("{}", 1u32 as f32);
  info2!("{}", 1u32 as f64);

  info2!("{}", 1i64 as int);
  info2!("{}", 1i64 as uint);
  info2!("{}", 1i64 as float);
  info2!("{}", 1i64 as *libc::FILE);
  info2!("{}", 1i64 as i8);
  info2!("{}", 1i64 as i16);
  info2!("{}", 1i64 as i32);
  info2!("{}", 1i64 as i64);
  info2!("{}", 1i64 as u8);
  info2!("{}", 1i64 as u16);
  info2!("{}", 1i64 as u32);
  info2!("{}", 1i64 as u64);
  info2!("{}", 1i64 as f32);
  info2!("{}", 1i64 as f64);

  info2!("{}", 1u64 as int);
  info2!("{}", 1u64 as uint);
  info2!("{}", 1u64 as float);
  info2!("{}", 1u64 as *libc::FILE);
  info2!("{}", 1u64 as i8);
  info2!("{}", 1u64 as i16);
  info2!("{}", 1u64 as i32);
  info2!("{}", 1u64 as i64);
  info2!("{}", 1u64 as u8);
  info2!("{}", 1u64 as u16);
  info2!("{}", 1u64 as u32);
  info2!("{}", 1u64 as u64);
  info2!("{}", 1u64 as f32);
  info2!("{}", 1u64 as f64);

  info2!("{}", 1u64 as int);
  info2!("{}", 1u64 as uint);
  info2!("{}", 1u64 as float);
  info2!("{}", 1u64 as *libc::FILE);
  info2!("{}", 1u64 as i8);
  info2!("{}", 1u64 as i16);
  info2!("{}", 1u64 as i32);
  info2!("{}", 1u64 as i64);
  info2!("{}", 1u64 as u8);
  info2!("{}", 1u64 as u16);
  info2!("{}", 1u64 as u32);
  info2!("{}", 1u64 as u64);
  info2!("{}", 1u64 as f32);
  info2!("{}", 1u64 as f64);

  info2!("{}", true as int);
  info2!("{}", true as uint);
  info2!("{}", true as float);
  info2!("{}", true as *libc::FILE);
  info2!("{}", true as i8);
  info2!("{}", true as i16);
  info2!("{}", true as i32);
  info2!("{}", true as i64);
  info2!("{}", true as u8);
  info2!("{}", true as u16);
  info2!("{}", true as u32);
  info2!("{}", true as u64);
  info2!("{}", true as f32);
  info2!("{}", true as f64);

  info2!("{}", 1. as int);
  info2!("{}", 1. as uint);
  info2!("{}", 1. as float);
  info2!("{}", 1. as i8);
  info2!("{}", 1. as i16);
  info2!("{}", 1. as i32);
  info2!("{}", 1. as i64);
  info2!("{}", 1. as u8);
  info2!("{}", 1. as u16);
  info2!("{}", 1. as u32);
  info2!("{}", 1. as u64);
  info2!("{}", 1. as f32);
  info2!("{}", 1. as f64);

  info2!("{}", 1f32 as int);
  info2!("{}", 1f32 as uint);
  info2!("{}", 1f32 as float);
  info2!("{}", 1f32 as i8);
  info2!("{}", 1f32 as i16);
  info2!("{}", 1f32 as i32);
  info2!("{}", 1f32 as i64);
  info2!("{}", 1f32 as u8);
  info2!("{}", 1f32 as u16);
  info2!("{}", 1f32 as u32);
  info2!("{}", 1f32 as u64);
  info2!("{}", 1f32 as f32);
  info2!("{}", 1f32 as f64);

  info2!("{}", 1f64 as int);
  info2!("{}", 1f64 as uint);
  info2!("{}", 1f64 as float);
  info2!("{}", 1f64 as i8);
  info2!("{}", 1f64 as i16);
  info2!("{}", 1f64 as i32);
  info2!("{}", 1f64 as i64);
  info2!("{}", 1f64 as u8);
  info2!("{}", 1f64 as u16);
  info2!("{}", 1f64 as u32);
  info2!("{}", 1f64 as u64);
  info2!("{}", 1f64 as f32);
  info2!("{}", 1f64 as f64);
}
