// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![cfg_attr(stage0, feature(custom_attribute))]
#![crate_name = "rustc_platform_intrinsics"]
#![unstable(feature = "rustc_private", issue = "27812")]
#![staged_api]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![feature(staged_api, rustc_private)]

extern crate rustc_llvm as llvm;
extern crate rustc;

use rustc::middle::ty;

pub struct Intrinsic {
    pub inputs: Vec<Type>,
    pub output: Type,

    pub definition: IntrinsicDef,
}

#[derive(Clone, Hash, Eq, PartialEq)]
pub enum Type {
    Integer(/* signed */ bool, u8),
    Float(u8),
    Pointer(Box<Type>),
    Vector(Box<Type>, u8),
    Aggregate(bool, Vec<Type>),
}

pub enum IntrinsicDef {
    Named(&'static str),
}

fn i(width: u8) -> Type { Type::Integer(true, width) }
fn u(width: u8) -> Type { Type::Integer(false, width) }
fn f(width: u8) -> Type { Type::Float(width) }
fn v(x: Type, length: u8) -> Type { Type::Vector(Box::new(x), length) }
fn agg(flatten: bool, types: Vec<Type>) -> Type {
    Type::Aggregate(flatten, types)
}

macro_rules! ty {
    (f32x8) => (v(f(32), 8));
    (f64x4) => (v(f(64), 4));

    (i8x32) => (v(i(8), 32));
    (i16x16) => (v(i(16), 16));
    (i32x8) => (v(i(32), 8));
    (i64x4) => (v(i(64), 4));

    (f32x4) => (v(f(32), 4));
    (f64x2) => (v(f(64), 2));

    (i8x16) => (v(i(8), 16));
    (i16x8) => (v(i(16), 8));
    (i32x4) => (v(i(32), 4));
    (i64x2) => (v(i(64), 2));

    (f32x2) => (v(f(32), 2));
    (i8x8) => (v(i(8), 8));
    (i16x4) => (v(i(16), 4));
    (i32x2) => (v(i(32), 2));
    (i64x1)=> (v(i(64), 1));

    (i64) => (i(64));
    (i32) => (i(32));
    (i16) => (i(16));
    (i8) => (i(8));
    (f32) => (f(32));
    (f64) => (f(64));
}
macro_rules! plain {
    ($name: expr, ($($inputs: tt),*) -> $output: tt) => {
        Intrinsic {
            inputs: vec![$(ty!($inputs)),*],
            output: ty!($output),
            definition: ::IntrinsicDef::Named($name)
        }
    }
}

mod x86;
mod arm;
mod aarch64;

impl Intrinsic {
    pub fn find<'tcx>(tcx: &ty::ctxt<'tcx>, name: &str) -> Option<Intrinsic> {
        if name.starts_with("x86_") {
            x86::find(tcx, name)
        } else if name.starts_with("arm_") {
            arm::find(tcx, name)
        } else if name.starts_with("aarch64_") {
            aarch64::find(tcx, name)
        } else {
            None
        }
    }
}
