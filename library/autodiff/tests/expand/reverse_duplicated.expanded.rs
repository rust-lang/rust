#[autodiff_into] fn squre(a : & Vec < f32 >, b : & mut f32,)
{ * b = a.into_iter().map(f32 :: square).sum() ; }
#[autodiff_into(Reverse, Const, Duplicated, Duplicated,)] fn
grad_squre(a : & Vec < f32 >, grad_a : & mut Vec < f32 >, b : & mut f32,
grad_b : & f32,)
{
    std :: hint :: black_box((squre(a, b,), grad_a, grad_b,)) ; unsafe
    { std :: mem :: zeroed() }
}
#![feature(prelude_import)]
#[prelude_import]
use std::prelude::rust_2018::*;
#[macro_use]
extern crate std;
use autodiff::autodiff;
#[autodiff_into]
fn squre(a: &Vec<f32>, b: &mut f32) {
    *b = a.into_iter().map(f32::square).sum();
}
#[autodiff_into(Reverse, Const, Duplicated, Duplicated)]
fn grad_squre(a: &Vec<f32>, grad_a: &mut Vec<f32>, b: &mut f32, grad_b: &f32) {
    std::hint::black_box((squre(a, b), grad_a, grad_b));
    unsafe { std::mem::zeroed() }
}
