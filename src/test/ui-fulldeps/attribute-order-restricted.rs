// aux-build:attr_proc_macro.rs
// compile-flags:--test

#![feature(test)]

extern crate test;
extern crate attr_proc_macro;
use attr_proc_macro::*;

#[attr_proc_macro] // OK
#[derive(Clone)]
struct Before;

#[derive(Clone)]
#[attr_proc_macro] //~ ERROR macro attributes must be placed before `#[derive]`
struct After;

#[attr_proc_macro] //~ ERROR macro attributes cannot be used together with `#[test]` or `#[bench]`
#[test]
fn test_before() {}

#[test]
#[attr_proc_macro] //~ ERROR macro attributes cannot be used together with `#[test]` or `#[bench]`
fn test_after() {}

#[attr_proc_macro] //~ ERROR macro attributes cannot be used together with `#[test]` or `#[bench]`
#[bench]
fn bench_before(b: &mut test::Bencher) {}

#[bench]
#[attr_proc_macro] //~ ERROR macro attributes cannot be used together with `#[test]` or `#[bench]`
fn bench_after(b: &mut test::Bencher) {}
