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
#![cfg_attr(stage0, staged_api)]
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
    Void,
    Integer(/* signed */ bool, u8, /* llvm width */ u8),
    Float(u8),
    Pointer(Box<Type>, Option<Box<Type>>, /* const */ bool),
    Vector(Box<Type>, Option<Box<Type>>, u8),
    Aggregate(bool, Vec<Type>),
}

pub enum IntrinsicDef {
    Named(&'static str),
}

fn i(width: u8) -> Type { Type::Integer(true, width, width) }
fn i_(width: u8, llvm_width: u8) -> Type { Type::Integer(true, width, llvm_width) }
fn u(width: u8) -> Type { Type::Integer(false, width, width) }
#[allow(dead_code)]
fn u_(width: u8, llvm_width: u8) -> Type { Type::Integer(false, width, llvm_width) }
fn f(width: u8) -> Type { Type::Float(width) }
fn v(x: Type, length: u8) -> Type { Type::Vector(Box::new(x), None, length) }
fn v_(x: Type, bitcast: Type, length: u8) -> Type {
    Type::Vector(Box::new(x), Some(Box::new(bitcast)), length)
}
fn agg(flatten: bool, types: Vec<Type>) -> Type {
    Type::Aggregate(flatten, types)
}
fn p(const_: bool, elem: Type, llvm_elem: Option<Type>) -> Type {
    Type::Pointer(Box::new(elem), llvm_elem.map(Box::new), const_)
}
fn void() -> Type {
    Type::Void
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
