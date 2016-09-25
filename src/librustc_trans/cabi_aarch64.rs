// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_upper_case_globals)]

use llvm::{Integer, Pointer, Float, Double, Struct, Array, Vector};
use abi::{self, FnType, ArgType};
use context::CrateContext;
use type_::Type;

fn ty_size(ty: Type) -> usize {
    abi::ty_size(ty, 8)
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
        Vector => match ty_size(ty) {
            4|8 => Some((ty, 1)),
            _   => None
        },
        _ => None
    };

    // Ensure we have at most four uniquely addressable members
    homog_agg.and_then(|(base_ty, members)| {
        if members > 0 && members <= 4 {
            Some((base_ty, members))
        } else {
            None
        }
    })
}

fn classify_ret_ty(ccx: &CrateContext, ret: &mut ArgType) {
    if is_reg_ty(ret.ty) {
        ret.extend_integer_width_to(32);
        return;
    }
    if let Some((base_ty, members)) = is_homogenous_aggregate_ty(ret.ty) {
        ret.cast = Some(Type::array(&base_ty, members));
        return;
    }
    let size = ty_size(ret.ty);
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
        ret.cast = Some(llty);
        return;
    }
    ret.make_indirect(ccx);
}

fn classify_arg_ty(ccx: &CrateContext, arg: &mut ArgType) {
    if is_reg_ty(arg.ty) {
        arg.extend_integer_width_to(32);
        return;
    }
    if let Some((base_ty, members)) = is_homogenous_aggregate_ty(arg.ty) {
        arg.cast = Some(Type::array(&base_ty, members));
        return;
    }
    let size = ty_size(arg.ty);
    if size <= 16 {
        let llty = if size == 0 {
            Type::array(&Type::i64(ccx), 0)
        } else if size == 1 {
            Type::i8(ccx)
        } else if size == 2 {
            Type::i16(ccx)
        } else if size <= 4 {
            Type::i32(ccx)
        } else if size <= 8 {
            Type::i64(ccx)
        } else {
            Type::array(&Type::i64(ccx), ((size + 7 ) / 8 ) as u64)
        };
        arg.cast = Some(llty);
        return;
    }
    arg.make_indirect(ccx);
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

pub fn compute_abi_info(ccx: &CrateContext, fty: &mut FnType) {
    if !fty.ret.is_ignore() {
        classify_ret_ty(ccx, &mut fty.ret);
    }

    for arg in &mut fty.args {
        if arg.is_ignore() { continue; }
        classify_arg_ty(ccx, arg);
    }
}
