#![feature(pattern_types, rustc_attrs)]
#![allow(incomplete_features)]

#[rustc_layout(debug)]
type X = std::num::NonZeroU32; //~ ERROR layout_of
#[rustc_layout(debug)]
type Y = u32 is 1..; //~ ERROR layout_of
#[rustc_layout(debug)]
type Z = Option<u32 is 1..>; //~ ERROR layout_of
#[rustc_layout(debug)]
type A = Option<std::num::NonZeroU32>; //~ ERROR layout_of

fn main() {
    let x: u32 is 1.. = unsafe { std::mem::transmute(42_u32) };
}
