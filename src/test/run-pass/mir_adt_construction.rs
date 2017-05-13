// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[repr(C)]
enum CEnum {
    Hello = 30,
    World = 60
}

fn test1(c: CEnum) -> i32 {
  let c2 = CEnum::Hello;
  match (c, c2) {
    (CEnum::Hello, CEnum::Hello) => 42,
    (CEnum::World, CEnum::Hello) => 0,
    _ => 1
  }
}

#[repr(packed)]
#[derive(PartialEq, Debug)]
struct Pakd {
    a: u64,
    b: u32,
    c: u16,
    d: u8,
    e: ()
}

impl Drop for Pakd {
    fn drop(&mut self) {}
}

fn test2() -> Pakd {
    Pakd { a: 42, b: 42, c: 42, d: 42, e: () }
}

#[derive(PartialEq, Debug)]
struct TupleLike(u64, u32);

fn test3() -> TupleLike {
    TupleLike(42, 42)
}

fn test4(x: fn(u64, u32) -> TupleLike) -> (TupleLike, TupleLike) {
    let y = TupleLike;
    (x(42, 84), y(42, 84))
}

fn test5(x: fn(u32) -> Option<u32>) -> (Option<u32>, Option<u32>) {
    let y = Some;
    (x(42), y(42))
}

fn main() {
  assert_eq!(test1(CEnum::Hello), 42);
  assert_eq!(test1(CEnum::World), 0);
  assert_eq!(test2(), Pakd { a: 42, b: 42, c: 42, d: 42, e: () });
  assert_eq!(test3(), TupleLike(42, 42));
  let t4 = test4(TupleLike);
  assert_eq!(t4.0, t4.1);
  let t5 = test5(Some);
  assert_eq!(t5.0, t5.1);
}
