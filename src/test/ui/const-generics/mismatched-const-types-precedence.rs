#![feature(generic_const_exprs)]
#![allow(incomplete_features)]


fn bar<const N: usize, const K: usize>(a: [u32; 2 + N], b: [u32; K]) {
  let _: [u32; 3 * 2 + N + K] = foo::<{ 2 + N }, K>(a, b);
  //~^ ERROR unconstrained generic constant
  //~| ERROR unconstrained generic constant
  //~| ERROR unconstrained generic constant
  //~| ERROR mismatched types
}

fn foo<const M: usize, const K: usize>(a: [u32; M], b: [u32; K]) -> [u32; 3 * {M + K}] {}
//~^ ERROR mismatched types


fn bar2<const N: usize, const K: usize>(a: [u32; 2 + N], b: [u32; K]) {
  let _: [u32; 3 * 2 + N * K] = foo2::<{ 2 + N }, K>(a, b);
  //~^ ERROR unconstrained generic constant
  //~| ERROR unconstrained generic constant
  //~| ERROR unconstrained generic constant
  //~| ERROR mismatched types
}

fn foo2<const M: usize, const K: usize>(a: [u32; M], b: [u32; K]) -> [u32; 3 * M * K] {}
//~^ ERROR mismatched types


fn bar3<const N: usize, const K: usize, const L: usize>(a: [u32; 2 + N], b: [u32; K + L]) {
  let _: [u32; 3 * 2 + N * K + L] = foo3::<{ 2 + N }, K, L>(a, b);
  //~^ ERROR unconstrained generic constant
  //~| ERROR unconstrained generic constant
  //~| ERROR unconstrained generic constant
  //~| ERROR mismatched types
}

fn foo3<const M: usize, const K: usize, const L: usize>(a: [u32; M], b: [u32; K + L])
  -> [u32; 3 * M * {K + L}] {}
//~^ ERROR mismatched types

fn main() {}
