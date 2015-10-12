// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use llvm::*;
use super::common::*;
use super::machine::*;
use trans::cabi::{ArgType, FnType};
use trans::type_::Type;

// Win64 ABI: http://msdn.microsoft.com/en-us/library/zthk2dkh.aspx

pub fn compute_abi_info(ccx: &CrateContext,
                          atys: &[Type],
                          rty: Type,
                          ret_def: bool) -> FnType {
    let mut arg_tys = Vec::new();

    let ret_ty;
    if !ret_def {
        ret_ty = ArgType::direct(Type::void(ccx), None, None, None);
    } else if rty.kind() == Struct {
        ret_ty = match llsize_of_alloc(ccx, rty) {
            1 => ArgType::direct(rty, Some(Type::i8(ccx)), None, None),
            2 => ArgType::direct(rty, Some(Type::i16(ccx)), None, None),
            4 => ArgType::direct(rty, Some(Type::i32(ccx)), None, None),
            8 => ArgType::direct(rty, Some(Type::i64(ccx)), None, None),
            _ => ArgType::indirect(rty, Some(Attribute::StructRet))
        };
    } else {
        let attr = if rty == Type::i1(ccx) { Some(Attribute::ZExt) } else { None };
        ret_ty = ArgType::direct(rty, None, None, attr);
    }

    for &t in atys {
        let ty = match t.kind() {
            Struct => {
                match llsize_of_alloc(ccx, t) {
                    1 => ArgType::direct(t, Some(Type::i8(ccx)), None, None),
                    2 => ArgType::direct(t, Some(Type::i16(ccx)), None, None),
                    4 => ArgType::direct(t, Some(Type::i32(ccx)), None, None),
                    8 => ArgType::direct(t, Some(Type::i64(ccx)), None, None),
                    _ => ArgType::indirect(t, None)
                }
            }
            _ => {
                let attr = if t == Type::i1(ccx) { Some(Attribute::ZExt) } else { None };
                ArgType::direct(t, None, None, attr)
            }
        };
        arg_tys.push(ty);
    }

    return FnType {
        arg_tys: arg_tys,
        ret_ty: ret_ty,
    };
}
