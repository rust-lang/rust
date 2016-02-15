// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_upper_case_globals)]

use llvm::{Struct, Array, Attribute};
use trans::cabi::{FnType, ArgType};
use trans::context::CrateContext;
use trans::type_::Type;

// Data layout: e-p:32:32-i64:64-v128:32:128-n32-S128

// See the https://github.com/kripken/emscripten-fastcomp-clang repository.
// The class `EmscriptenABIInfo` in `/lib/CodeGen/TargetInfo.cpp` contains the ABI definitions.

fn classify_ret_ty(ccx: &CrateContext, ty: Type) -> ArgType {
    match ty.kind() {
        Struct => {
            let field_types = ty.field_types();
            if field_types.len() == 1 {
                ArgType::direct(ty, Some(field_types[0]), None, None)
            } else {
                ArgType::indirect(ty, Some(Attribute::StructRet))
            }
        },
        Array => {
            ArgType::indirect(ty, Some(Attribute::StructRet))
        },
        _ => {
            let attr = if ty == Type::i1(ccx) { Some(Attribute::ZExt) } else { None };
            ArgType::direct(ty, None, None, attr)
        }
    }
}

fn classify_arg_ty(ccx: &CrateContext, ty: Type) -> ArgType {
    if ty.is_aggregate() {
        ArgType::indirect(ty, Some(Attribute::ByVal))
    } else {
        let attr = if ty == Type::i1(ccx) { Some(Attribute::ZExt) } else { None };
        ArgType::direct(ty, None, None, attr)
    }
}

pub fn compute_abi_info(ccx: &CrateContext,
                        atys: &[Type],
                        rty: Type,
                        ret_def: bool) -> FnType {
    let mut arg_tys = Vec::new();
    for &aty in atys {
        let ty = classify_arg_ty(ccx, aty);
        arg_tys.push(ty);
    }

    let ret_ty = if ret_def {
        classify_ret_ty(ccx, rty)
    } else {
        ArgType::direct(Type::void(ccx), None, None, None)
    };

    return FnType {
        arg_tys: arg_tys,
        ret_ty: ret_ty,
    };
}
