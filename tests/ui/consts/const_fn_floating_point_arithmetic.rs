// gate-test-const_fn_floating_point_arithmetic

//@ revisions: stock gated

#![feature(rustc_attrs)]
#![cfg_attr(gated, feature(const_fn_floating_point_arithmetic))]

const fn add(f: f32) -> f32 { f + 2.0 }
//[stock]~^ floating point arithmetic
const fn sub(f: f32) -> f32 { 2.0 - f }
//[stock]~^ floating point arithmetic
const fn mul(f: f32, g: f32) -> f32 { f * g }
//[stock]~^ floating point arithmetic
const fn div(f: f32, g: f32) -> f32 { f / g }
//[stock]~^ floating point arithmetic
const fn neg(f: f32) -> f32 { -f }
//[stock]~^ floating point arithmetic

#[rustc_error]
fn main() {} //[gated]~ fatal error triggered by #[rustc_error]
