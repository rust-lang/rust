#[autodiff_into] fn array(arr : & [[[f32 ; 2] ; 2] ; 2],) -> f32
{ arr [0] [0] [0] * arr [1] [1] [1] }
#[autodiff_into(Reverse, Active, Duplicated,)] fn
grad_array(arr : & [[[f32 ; 2] ; 2] ; 2], grad_arr : & mut
[[[f32 ; 2] ; 2] ; 2], tang_y : f32,)
{
    std :: hint :: black_box((array(arr,), grad_arr, tang_y,)) ; unsafe
    { std :: mem :: zeroed() }
}
#![feature(prelude_import)]
#[prelude_import]
use std::prelude::rust_2018::*;
#[macro_use]
extern crate std;
use autodiff::autodiff;
#[autodiff_into]
fn array(arr: &[[[f32; 2]; 2]; 2]) -> f32 {
    arr[0][0][0] * arr[1][1][1]
}
#[autodiff_into(Reverse, Active, Duplicated)]
fn grad_array(arr: &[[[f32; 2]; 2]; 2], grad_arr: &mut [[[f32; 2]; 2]; 2], tang_y: f32) {
    std::hint::black_box((array(arr), grad_arr, tang_y));
    unsafe { std::mem::zeroed() }
}
