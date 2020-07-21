// run-pass
#![allow(dead_code)]
#![feature(const_fn_pointer)]
#![feature(const_fn)]


const fn foo() {}
const FOO: const fn() = foo;
const fn bar() { FOO() }
const fn baz(x: const fn()) { x() }
const fn bazz() { baz(FOO) }

trait Bar { const F: const fn(Self) -> Self; }

const fn map_i32(x: i32) -> i32 { x * 2 }
impl Bar for i32 { const F: const fn(Self) -> Self = map_i32; } 
const fn map_u32(x: u32) -> u32 { x * 3 }
impl Bar for u32 { const F: const fn(Self) -> Self = map_u32; } 

const fn map_smth<T: Bar>(v: T) -> T {
  <T as Bar>::F(v)
}

fn main() { 
  const VAR: i32 = map_smth(2);
  assert_eq!(VAR, 4);
}
