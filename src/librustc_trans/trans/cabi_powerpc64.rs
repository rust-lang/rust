// Copyright 2014-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME: The PowerPC64 ABI needs to zero or sign extend function
// call parameters, but compute_abi_info() is passed LLVM types
// which have no sign information.
//
// Alignment of 128 bit types is not currently handled, this will
// need to be fixed when PowerPC vector support is added.

use llvm::{Integer, Pointer, Float, Double, Struct, Array, Attribute};
use trans::cabi::{FnType, ArgType};
use trans::context::CrateContext;
use trans::type_::Type;

use std::cmp;

fn align_up_to(off: usize, a: usize) -> usize {
    return (off + a - 1) / a * a;
}

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
        _ => panic!("ty_align: unhandled type")
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
        _ => panic!("ty_size: unhandled type")
    }
}

fn is_homogenous_aggregate_ty(ty: Type) -> Option<(Type, u64)> {
    fn check_array(ty: Type) -> Option<(Type, u64)> {
        let len = ty.array_length() as u64;
        if len == 0 {
            return None
        }
        let elt = ty.element_type();

        // if our element is an HFA/HVA, so are we; multiply members by our len
        is_homogenous_aggregate_ty(elt).map(|(base_ty, members)| (base_ty, len * members))
    }

    fn check_struct(ty: Type) -> Option<(Type, u64)> {
        let str_tys = ty.field_types();
        if str_tys.len() == 0 {
            return None
        }

        let mut prev_base_ty = None;
        let mut members = 0;
        for opt_homog_agg in str_tys.iter().map(|t| is_homogenous_aggregate_ty(*t)) {
            match (prev_base_ty, opt_homog_agg) {
                // field isn't itself an HFA, so we aren't either
                (_, None) => return None,

                // first field - store its type and number of members
                (None, Some((field_ty, field_members))) => {
                    prev_base_ty = Some(field_ty);
                    members = field_members;
                },

                // 2nd or later field - give up if it's a different type; otherwise incr. members
                (Some(prev_ty), Some((field_ty, field_members))) => {
                    if prev_ty != field_ty {
                        return None;
                    }
                    members += field_members;
                }
            }
        }

        // Because of previous checks, we know prev_base_ty is Some(...) because
        //   1. str_tys has at least one element; and
        //   2. prev_base_ty was filled in (or we would've returned early)
        let (base_ty, members) = (prev_base_ty.unwrap(), members);

        // Ensure there is no padding.
        if ty_size(ty) == ty_size(base_ty) * (members as usize) {
            Some((base_ty, members))
        } else {
            None
        }
    }

    let homog_agg = match ty.kind() {
        Float  => Some((ty, 1)),
        Double => Some((ty, 1)),
        Array  => check_array(ty),
        Struct => check_struct(ty),
        _ => None
    };

    // Ensure we have at most eight uniquely addressable members
    homog_agg.and_then(|(base_ty, members)| {
        if members > 0 && members <= 8 {
            Some((base_ty, members))
        } else {
            None
        }
    })
}

fn classify_ret_ty(ccx: &CrateContext, ty: Type) -> ArgType {
    if is_reg_ty(ty) {
        let attr = if ty == Type::i1(ccx) { Some(Attribute::ZExt) } else { None };
        return ArgType::direct(ty, None, None, attr);
    }

    // The PowerPC64 big endian ABI doesn't return aggregates in registers
    if ccx.sess().target.target.arch == "powerpc64" {
        return ArgType::indirect(ty, Some(Attribute::StructRet))
    }

    if let Some((base_ty, members)) = is_homogenous_aggregate_ty(ty) {
        let llty = Type::array(&base_ty, members);
        return ArgType::direct(ty, Some(llty), None, None);
    }
    let size = ty_size(ty);
    if size <= 16 {
        let llty = if size <= 1 {
            Type::i8(ccx)
        } else if size <= 2 {
            Type::i16(ccx)
        } else if size <= 4 {
            Type::i32(ccx)
        } else if size <= 8 {
            Type::i64(ccx)
        } else {
            Type::array(&Type::i64(ccx), ((size + 7 ) / 8 ) as u64)
        };
        return ArgType::direct(ty, Some(llty), None, None);
    }

    ArgType::indirect(ty, Some(Attribute::StructRet))
}

fn classify_arg_ty(ccx: &CrateContext, ty: Type) -> ArgType {
    if is_reg_ty(ty) {
        let attr = if ty == Type::i1(ccx) { Some(Attribute::ZExt) } else { None };
        return ArgType::direct(ty, None, None, attr);
    }
    if let Some((base_ty, members)) = is_homogenous_aggregate_ty(ty) {
        let llty = Type::array(&base_ty, members);
        return ArgType::direct(ty, Some(llty), None, None);
    }

    ArgType::direct(
        ty,
        Some(struct_ty(ccx, ty)),
        None,
        None
    )
}

fn is_reg_ty(ty: Type) -> bool {
    match ty.kind() {
        Integer
        | Pointer
        | Float
        | Double => true,
        _ => false
    }
}

fn coerce_to_long(ccx: &CrateContext, size: usize) -> Vec<Type> {
    let long_ty = Type::i64(ccx);
    let mut args = Vec::new();

    let mut n = size / 64;
    while n > 0 {
        args.push(long_ty);
        n -= 1;
    }

    let r = size % 64;
    if r > 0 {
        args.push(Type::ix(ccx, r as u64));
    }

    args
}

fn struct_ty(ccx: &CrateContext, ty: Type) -> Type {
    let size = ty_size(ty) * 8;
    Type::struct_(ccx, &coerce_to_long(ccx, size), false)
}

pub fn compute_abi_info(ccx: &CrateContext,
                        atys: &[Type],
                        rty: Type,
                        ret_def: bool) -> FnType {
    let ret_ty = if ret_def {
        classify_ret_ty(ccx, rty)
    } else {
        ArgType::direct(Type::void(ccx), None, None, None)
    };

    let mut arg_tys = Vec::new();
    for &aty in atys {
        let ty = classify_arg_ty(ccx, aty);
        arg_tys.push(ty);
    };

    return FnType {
        arg_tys: arg_tys,
        ret_ty: ret_ty,
    };
}
