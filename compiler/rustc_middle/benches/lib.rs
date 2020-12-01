#![feature(test)]

extern crate test;

use test::Bencher;

// Static/dynamic method dispatch

struct Struct {
    field: isize,
}

trait Trait {
    fn method(&self) -> isize;
}

impl Trait for Struct {
    fn method(&self) -> isize {
        self.field
    }
}

#[bench]
fn trait_vtable_method_call(b: &mut Bencher) {
    let s = Struct { field: 10 };
    let t = &s as &dyn Trait;
    b.iter(|| t.method());
}

#[bench]
fn trait_static_method_call(b: &mut Bencher) {
    let s = Struct { field: 10 };
    b.iter(|| s.method());
}

// Overhead of various match forms

#[bench]
fn option_some(b: &mut Bencher) {
    let x = Some(10);
    b.iter(|| match x {
        Some(y) => y,
        None => 11,
    });
}

#[bench]
fn vec_pattern(b: &mut Bencher) {
    let x = [1, 2, 3, 4, 5, 6];
    b.iter(|| match x {
        [1, 2, 3, ..] => 10,
        _ => 11,
    });
}
