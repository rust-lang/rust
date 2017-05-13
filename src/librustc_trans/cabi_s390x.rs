// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME: The assumes we're using the non-vector ABI, i.e. compiling
// for a pre-z13 machine or using -mno-vx.

use llvm::{Integer, Pointer, Float, Double, Struct, Array, Vector};
use abi::{align_up_to, FnType, ArgType};
use context::CrateContext;
use type_::Type;

use std::cmp;

fn align(off: usize, ty: Type) -> usize {
    let a = ty_align(ty);
    return align_up_to(off, a);
}

fn ty_align(ty: Type) -> usize {
    match ty.kind() {
        Integer => ((ty.int_width() as usize) + 7) / 8,
        Pointer => 8,
        Float => 4,
        Double => 8,
        Struct => {
            if ty.is_packed() {
                1
            } else {
                let str_tys = ty.field_types();
                str_tys.iter().fold(1, |a, t| cmp::max(a, ty_align(*t)))
            }
        }
        Array => {
            let elt = ty.element_type();
            ty_align(elt)
        }
        Vector => ty_size(ty),
        _ => bug!("ty_align: unhandled type")
    }
}

fn ty_size(ty: Type) -> usize {
    match ty.kind() {
        Integer => ((ty.int_width() as usize) + 7) / 8,
        Pointer => 8,
        Float => 4,
        Double => 8,
        Struct => {
            if ty.is_packed() {
                let str_tys = ty.field_types();
                str_tys.iter().fold(0, |s, t| s + ty_size(*t))
            } else {
                let str_tys = ty.field_types();
                let size = str_tys.iter().fold(0, |s, t| align(s, *t) + ty_size(*t));
                align(size, ty)
            }
        }
        Array => {
            let len = ty.array_length();
            let elt = ty.element_type();
            let eltsz = ty_size(elt);
            len * eltsz
        }
        Vector => {
            let len = ty.vector_length();
            let elt = ty.element_type();
            let eltsz = ty_size(elt);
            len * eltsz
        }
        _ => bug!("ty_size: unhandled type")
    }
}

fn classify_ret_ty(ccx: &CrateContext, ret: &mut ArgType) {
    if is_reg_ty(ret.ty) {
        ret.extend_integer_width_to(64);
    } else {
        ret.make_indirect(ccx);
    }
}

fn classify_arg_ty(ccx: &CrateContext, arg: &mut ArgType) {
    if arg.ty.kind() == Struct {
        fn is_single_fp_element(tys: &[Type]) -> bool {
            if tys.len() != 1 {
                return false;
            }
            match tys[0].kind() {
                Float | Double => true,
                Struct => is_single_fp_element(&tys[0].field_types()),
                _ => false
            }
        }

        if is_single_fp_element(&arg.ty.field_types()) {
            match ty_size(arg.ty) {
                4 => arg.cast = Some(Type::f32(ccx)),
                8 => arg.cast = Some(Type::f64(ccx)),
                _ => arg.make_indirect(ccx)
            }
        } else {
            match ty_size(arg.ty) {
                1 => arg.cast = Some(Type::i8(ccx)),
                2 => arg.cast = Some(Type::i16(ccx)),
                4 => arg.cast = Some(Type::i32(ccx)),
                8 => arg.cast = Some(Type::i64(ccx)),
                _ => arg.make_indirect(ccx)
            }
        }
        return;
    }

    if is_reg_ty(arg.ty) {
        arg.extend_integer_width_to(64);
    } else {
        arg.make_indirect(ccx);
    }
}

fn is_reg_ty(ty: Type) -> bool {
    match ty.kind() {
        Integer
        | Pointer
        | Float
        | Double => ty_size(ty) <= 8,
        _ => false
    }
}

pub fn compute_abi_info(ccx: &CrateContext, fty: &mut FnType) {
    if !fty.ret.is_ignore() {
        classify_ret_ty(ccx, &mut fty.ret);
    }

    for arg in &mut fty.args {
        if arg.is_ignore() { continue; }
        classify_arg_ty(ccx, arg);
    }
}
