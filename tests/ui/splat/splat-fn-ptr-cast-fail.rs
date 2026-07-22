//! Test casting splatted functions to non-splatted function pointers fails.

#![allow(incomplete_features)]
#![feature(splat, tuple_trait)]

use std::marker::Tuple;

fn tuple_args(#[splat] (_a, _b): (u32, i8)) {}

fn splat_non_terminal_arg(#[splat] (_a, _b): (u32, i8), _c: f64) {}

fn f<Args: Tuple>(#[splat] args: Args) {}

fn main() {
    // Function pointers
    let _fn_ptr: fn((u32, i8)) = tuple_args; //~ ERROR mismatched types
    let _fn_ptr: fn((u32, i8), f64) = splat_non_terminal_arg; //~ ERROR mismatched types

    let _fn_ptr: fn((u32, i8)) = tuple_args as fn((u32, i8)); //~ ERROR non-primitive cast
    let _fn_ptr: fn((u32, i8), f64) = splat_non_terminal_arg as fn((u32, i8), f64); //~ ERROR non-primitive cast

    // Bug #158603 regression test variants
    const _F2: fn((u8, u32)) = f::<(u8, u32)>; //~ ERROR mismatched types
    const _F1: fn(((u8, u32),)) = f::<((u8, u32),)>; //~ ERROR mismatched types
}
