// Regression test for #84170: a `const fn` returning `Vec<Box<dyn Any>>` was rejected
// with E0723 "trait bounds other than `Sized` on const fn parameters are unstable" even
// though no trait object value is created.
//@ check-pass

#![allow(dead_code)]

const fn newv() -> Vec<Box<dyn std::any::Any>> {
    Vec::new()
}

const fn new(_val: &Vec<Box<dyn std::any::Any>>) {}

fn main() {}
