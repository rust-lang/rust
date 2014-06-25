// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate libc;

pub fn main() {
  let f = 1 as *const libc::FILE;
  println!("{}", f as int);
  println!("{}", f as uint);
  println!("{}", f as i8);
  println!("{}", f as i16);
  println!("{}", f as i32);
  println!("{}", f as i64);
  println!("{}", f as u8);
  println!("{}", f as u16);
  println!("{}", f as u32);
  println!("{}", f as u64);

  println!("{}", 1 as int);
  println!("{}", 1 as uint);
  println!("{}", 1 as *const libc::FILE);
  println!("{}", 1 as i8);
  println!("{}", 1 as i16);
  println!("{}", 1 as i32);
  println!("{}", 1 as i64);
  println!("{}", 1 as u8);
  println!("{}", 1 as u16);
  println!("{}", 1 as u32);
  println!("{}", 1 as u64);
  println!("{}", 1 as f32);
  println!("{}", 1 as f64);

  println!("{}", 1u as int);
  println!("{}", 1u as uint);
  println!("{}", 1u as *const libc::FILE);
  println!("{}", 1u as i8);
  println!("{}", 1u as i16);
  println!("{}", 1u as i32);
  println!("{}", 1u as i64);
  println!("{}", 1u as u8);
  println!("{}", 1u as u16);
  println!("{}", 1u as u32);
  println!("{}", 1u as u64);
  println!("{}", 1u as f32);
  println!("{}", 1u as f64);

  println!("{}", 1i8 as int);
  println!("{}", 1i8 as uint);
  println!("{}", 1i8 as *const libc::FILE);
  println!("{}", 1i8 as i8);
  println!("{}", 1i8 as i16);
  println!("{}", 1i8 as i32);
  println!("{}", 1i8 as i64);
  println!("{}", 1i8 as u8);
  println!("{}", 1i8 as u16);
  println!("{}", 1i8 as u32);
  println!("{}", 1i8 as u64);
  println!("{}", 1i8 as f32);
  println!("{}", 1i8 as f64);

  println!("{}", 1u8 as int);
  println!("{}", 1u8 as uint);
  println!("{}", 1u8 as *const libc::FILE);
  println!("{}", 1u8 as i8);
  println!("{}", 1u8 as i16);
  println!("{}", 1u8 as i32);
  println!("{}", 1u8 as i64);
  println!("{}", 1u8 as u8);
  println!("{}", 1u8 as u16);
  println!("{}", 1u8 as u32);
  println!("{}", 1u8 as u64);
  println!("{}", 1u8 as f32);
  println!("{}", 1u8 as f64);

  println!("{}", 1i16 as int);
  println!("{}", 1i16 as uint);
  println!("{}", 1i16 as *const libc::FILE);
  println!("{}", 1i16 as i8);
  println!("{}", 1i16 as i16);
  println!("{}", 1i16 as i32);
  println!("{}", 1i16 as i64);
  println!("{}", 1i16 as u8);
  println!("{}", 1i16 as u16);
  println!("{}", 1i16 as u32);
  println!("{}", 1i16 as u64);
  println!("{}", 1i16 as f32);
  println!("{}", 1i16 as f64);

  println!("{}", 1u16 as int);
  println!("{}", 1u16 as uint);
  println!("{}", 1u16 as *const libc::FILE);
  println!("{}", 1u16 as i8);
  println!("{}", 1u16 as i16);
  println!("{}", 1u16 as i32);
  println!("{}", 1u16 as i64);
  println!("{}", 1u16 as u8);
  println!("{}", 1u16 as u16);
  println!("{}", 1u16 as u32);
  println!("{}", 1u16 as u64);
  println!("{}", 1u16 as f32);
  println!("{}", 1u16 as f64);

  println!("{}", 1i32 as int);
  println!("{}", 1i32 as uint);
  println!("{}", 1i32 as *const libc::FILE);
  println!("{}", 1i32 as i8);
  println!("{}", 1i32 as i16);
  println!("{}", 1i32 as i32);
  println!("{}", 1i32 as i64);
  println!("{}", 1i32 as u8);
  println!("{}", 1i32 as u16);
  println!("{}", 1i32 as u32);
  println!("{}", 1i32 as u64);
  println!("{}", 1i32 as f32);
  println!("{}", 1i32 as f64);

  println!("{}", 1u32 as int);
  println!("{}", 1u32 as uint);
  println!("{}", 1u32 as *const libc::FILE);
  println!("{}", 1u32 as i8);
  println!("{}", 1u32 as i16);
  println!("{}", 1u32 as i32);
  println!("{}", 1u32 as i64);
  println!("{}", 1u32 as u8);
  println!("{}", 1u32 as u16);
  println!("{}", 1u32 as u32);
  println!("{}", 1u32 as u64);
  println!("{}", 1u32 as f32);
  println!("{}", 1u32 as f64);

  println!("{}", 1i64 as int);
  println!("{}", 1i64 as uint);
  println!("{}", 1i64 as *const libc::FILE);
  println!("{}", 1i64 as i8);
  println!("{}", 1i64 as i16);
  println!("{}", 1i64 as i32);
  println!("{}", 1i64 as i64);
  println!("{}", 1i64 as u8);
  println!("{}", 1i64 as u16);
  println!("{}", 1i64 as u32);
  println!("{}", 1i64 as u64);
  println!("{}", 1i64 as f32);
  println!("{}", 1i64 as f64);

  println!("{}", 1u64 as int);
  println!("{}", 1u64 as uint);
  println!("{}", 1u64 as *const libc::FILE);
  println!("{}", 1u64 as i8);
  println!("{}", 1u64 as i16);
  println!("{}", 1u64 as i32);
  println!("{}", 1u64 as i64);
  println!("{}", 1u64 as u8);
  println!("{}", 1u64 as u16);
  println!("{}", 1u64 as u32);
  println!("{}", 1u64 as u64);
  println!("{}", 1u64 as f32);
  println!("{}", 1u64 as f64);

  println!("{}", 1u64 as int);
  println!("{}", 1u64 as uint);
  println!("{}", 1u64 as *const libc::FILE);
  println!("{}", 1u64 as i8);
  println!("{}", 1u64 as i16);
  println!("{}", 1u64 as i32);
  println!("{}", 1u64 as i64);
  println!("{}", 1u64 as u8);
  println!("{}", 1u64 as u16);
  println!("{}", 1u64 as u32);
  println!("{}", 1u64 as u64);
  println!("{}", 1u64 as f32);
  println!("{}", 1u64 as f64);

  println!("{}", true as int);
  println!("{}", true as uint);
  println!("{}", true as *const libc::FILE);
  println!("{}", true as i8);
  println!("{}", true as i16);
  println!("{}", true as i32);
  println!("{}", true as i64);
  println!("{}", true as u8);
  println!("{}", true as u16);
  println!("{}", true as u32);
  println!("{}", true as u64);
  println!("{}", true as f32);
  println!("{}", true as f64);

  println!("{}", 1. as int);
  println!("{}", 1. as uint);
  println!("{}", 1. as i8);
  println!("{}", 1. as i16);
  println!("{}", 1. as i32);
  println!("{}", 1. as i64);
  println!("{}", 1. as u8);
  println!("{}", 1. as u16);
  println!("{}", 1. as u32);
  println!("{}", 1. as u64);
  println!("{}", 1. as f32);
  println!("{}", 1. as f64);

  println!("{}", 1f32 as int);
  println!("{}", 1f32 as uint);
  println!("{}", 1f32 as i8);
  println!("{}", 1f32 as i16);
  println!("{}", 1f32 as i32);
  println!("{}", 1f32 as i64);
  println!("{}", 1f32 as u8);
  println!("{}", 1f32 as u16);
  println!("{}", 1f32 as u32);
  println!("{}", 1f32 as u64);
  println!("{}", 1f32 as f32);
  println!("{}", 1f32 as f64);

  println!("{}", 1f64 as int);
  println!("{}", 1f64 as uint);
  println!("{}", 1f64 as i8);
  println!("{}", 1f64 as i16);
  println!("{}", 1f64 as i32);
  println!("{}", 1f64 as i64);
  println!("{}", 1f64 as u8);
  println!("{}", 1f64 as u16);
  println!("{}", 1f64 as u32);
  println!("{}", 1f64 as u64);
  println!("{}", 1f64 as f32);
  println!("{}", 1f64 as f64);
}
