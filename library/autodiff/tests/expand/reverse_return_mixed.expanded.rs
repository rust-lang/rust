#[autodiff_into] fn sqrt(a : f32, b : & f32, c : & f32, d : f32,) -> f32
{ a * (b * b + c * c * d * d).sqrt() }
#[autodiff_into(Reverse, Active, Active, Duplicated, Const, Active,)] fn
grad_sqrt(a : f32, b : & f32, grad_b : & mut f32, c : & f32, d : f32, tang_y :
f32,) -> (f32, f32,)
{
    std :: hint :: black_box((sqrt(a, b, c, d,), grad_b, tang_y,)) ; unsafe
    { std :: mem :: zeroed() }
}
#![feature(prelude_import)]
#[prelude_import]
use std::prelude::rust_2018::*;
#[macro_use]
extern crate std;
use autodiff::autodiff;
#[autodiff_into]
fn sqrt(a: f32, b: &f32, c: &f32, d: f32) -> f32 {
    a * (b * b + c * c * d * d).sqrt()
}
#[autodiff_into(Reverse, Active, Active, Duplicated, Const, Active)]
fn grad_sqrt(a: f32, b: &f32, grad_b: &mut f32, c: &f32, d: f32, tang_y: f32) -> (f32, f32) {
    std::hint::black_box((sqrt(a, b, c, d), grad_b, tang_y));
    unsafe { std::mem::zeroed() }
}
