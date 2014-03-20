// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use syntax::abi::{OsWin32, OsMacos};
use lib::llvm::*;
use super::cabi::*;
use super::common::*;
use super::machine::*;
use middle::trans::type_::Type;

pub fn compute_abi_info(ccx: &CrateContext,
                        atys: &[Type],
                        rty: Type,
                        ret_def: bool) -> FnType {
    let mut arg_tys = Vec::new();

    let ret_ty;
    if !ret_def {
        ret_ty = ArgType::direct(Type::void(ccx), None, None, None);
    } else if rty.kind() == Struct {
        // Returning a structure. Most often, this will use
        // a hidden first argument. On some platforms, though,
        // small structs are returned as integers.
        //
        // Some links:
        // http://www.angelcode.com/dev/callconv/callconv.html
        // Clang's ABI handling is in lib/CodeGen/TargetInfo.cpp

        enum Strategy { RetValue(Type), RetPointer }
        let strategy = match ccx.sess().targ_cfg.os {
            OsWin32 | OsMacos => {
                match llsize_of_alloc(ccx, rty) {
                    1 => RetValue(Type::i8(ccx)),
                    2 => RetValue(Type::i16(ccx)),
                    4 => RetValue(Type::i32(ccx)),
                    8 => RetValue(Type::i64(ccx)),
                    _ => RetPointer
                }
            }
            _ => {
                RetPointer
            }
        };

        match strategy {
            RetValue(t) => {
                ret_ty = ArgType::direct(rty, Some(t), None, None);
            }
            RetPointer => {
                ret_ty = ArgType::indirect(rty, Some(StructRetAttribute));
            }
        }
    } else {
        ret_ty = ArgType::direct(rty, None, None, None);
    }

    for &t in atys.iter() {
        let ty = match t.kind() {
            Struct => {
                let size = llsize_of_alloc(ccx, t);
                if size == 0 {
                    ArgType::ignore(t)
                } else {
                    ArgType::indirect(t, Some(ByValAttribute))
                }
            }
            _ => ArgType::direct(t, None, None, None),
        };
        arg_tys.push(ty);
    }

    return FnType {
        arg_tys: arg_tys,
        ret_ty: ret_ty,
    };
}
