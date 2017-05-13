// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use llvm::{Integer, Pointer, Float, Double, Struct, Array, Vector};
use abi::{self, align_up_to, FnType, ArgType};
use context::CrateContext;
use type_::Type;

use std::cmp;

pub enum Flavor {
    General,
    Ios
}

type TyAlignFn = fn(ty: Type) -> usize;

fn align(off: usize, ty: Type, align_fn: TyAlignFn) -> usize {
    let a = align_fn(ty);
    return align_up_to(off, a);
}

fn general_ty_align(ty: Type) -> usize {
    abi::ty_align(ty, 4)
}

// For more information see:
// ARMv7
// https://developer.apple.com/library/ios/documentation/Xcode/Conceptual
//    /iPhoneOSABIReference/Articles/ARMv7FunctionCallingConventions.html
// ARMv6
// https://developer.apple.com/library/ios/documentation/Xcode/Conceptual
//    /iPhoneOSABIReference/Articles/ARMv6FunctionCallingConventions.html
fn ios_ty_align(ty: Type) -> usize {
    match ty.kind() {
        Integer => cmp::min(4, ((ty.int_width() as usize) + 7) / 8),
        Pointer => 4,
        Float => 4,
        Double => 4,
        Struct => {
            if ty.is_packed() {
                1
            } else {
                let str_tys = ty.field_types();
                str_tys.iter().fold(1, |a, t| cmp::max(a, ios_ty_align(*t)))
            }
        }
        Array => {
            let elt = ty.element_type();
            ios_ty_align(elt)
        }
        Vector => {
            let len = ty.vector_length();
            let elt = ty.element_type();
            ios_ty_align(elt) * len
        }
        _ => bug!("ty_align: unhandled type")
    }
}

fn ty_size(ty: Type, align_fn: TyAlignFn) -> usize {
    match ty.kind() {
        Integer => ((ty.int_width() as usize) + 7) / 8,
        Pointer => 4,
        Float => 4,
        Double => 8,
        Struct => {
            if ty.is_packed() {
                let str_tys = ty.field_types();
                str_tys.iter().fold(0, |s, t| s + ty_size(*t, align_fn))
            } else {
                let str_tys = ty.field_types();
                let size = str_tys.iter()
                                  .fold(0, |s, t| {
                                      align(s, *t, align_fn) + ty_size(*t, align_fn)
                                  });
                align(size, ty, align_fn)
            }
        }
        Array => {
            let len = ty.array_length();
            let elt = ty.element_type();
            let eltsz = ty_size(elt, align_fn);
            len * eltsz
        }
        Vector => {
            let len = ty.vector_length();
            let elt = ty.element_type();
            let eltsz = ty_size(elt, align_fn);
            len * eltsz
        }
        _ => bug!("ty_size: unhandled type")
    }
}

fn classify_ret_ty(ccx: &CrateContext, ret: &mut ArgType, align_fn: TyAlignFn) {
    if is_reg_ty(ret.ty) {
        ret.extend_integer_width_to(32);
        return;
    }
    let size = ty_size(ret.ty, align_fn);
    if size <= 4 {
        let llty = if size <= 1 {
            Type::i8(ccx)
        } else if size <= 2 {
            Type::i16(ccx)
        } else {
            Type::i32(ccx)
        };
        ret.cast = Some(llty);
        return;
    }
    ret.make_indirect(ccx);
}

fn classify_arg_ty(ccx: &CrateContext, arg: &mut ArgType, align_fn: TyAlignFn) {
    if is_reg_ty(arg.ty) {
        arg.extend_integer_width_to(32);
        return;
    }
    let align = align_fn(arg.ty);
    let size = ty_size(arg.ty, align_fn);
    let llty = if align <= 4 {
        Type::array(&Type::i32(ccx), ((size + 3) / 4) as u64)
    } else {
        Type::array(&Type::i64(ccx), ((size + 7) / 8) as u64)
    };
    arg.cast = Some(llty);
}

fn is_reg_ty(ty: Type) -> bool {
    match ty.kind() {
        Integer
        | Pointer
        | Float
        | Double
        | Vector => true,
        _ => false
    }
}

pub fn compute_abi_info(ccx: &CrateContext, fty: &mut FnType, flavor: Flavor) {
    let align_fn = match flavor {
        Flavor::General => general_ty_align as TyAlignFn,
        Flavor::Ios => ios_ty_align as TyAlignFn,
    };

    if !fty.ret.is_ignore() {
        classify_ret_ty(ccx, &mut fty.ret, align_fn);
    }

    for arg in &mut fty.args {
        if arg.is_ignore() { continue; }
        classify_arg_ty(ccx, arg, align_fn);
    }
}
