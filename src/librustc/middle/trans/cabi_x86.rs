// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use driver::session::{os_win32, os_macos};
use lib::llvm::*;
use super::cabi::*;
use super::common::*;
use super::machine::*;
use middle::trans::type_::Type;

pub fn compute_abi_info(ccx: &mut CrateContext,
                        atys: &[Type],
                        rty: Type,
                        ret_def: bool) -> FnType {
    let mut arg_tys = ~[];
    let mut attrs = ~[];

    let ret_ty;
    let sret;
    if !ret_def {
        ret_ty = LLVMType {
            cast: false,
            ty: Type::void(),
        };
        sret = false;
    } else if rty.kind() == Struct {
        // Returning a structure. Most often, this will use
        // a hidden first argument. On some platforms, though,
        // small structs are returned as integers.
        //
        // Some links:
        // http://www.angelcode.com/dev/callconv/callconv.html
        // Clang's ABI handling is in lib/CodeGen/TargetInfo.cpp

        enum Strategy { RetValue(Type), RetPointer }
        let strategy = match ccx.sess.targ_cfg.os {
            os_win32 | os_macos => {
                match llsize_of_alloc(ccx, rty) {
                    1 => RetValue(Type::i8()),
                    2 => RetValue(Type::i16()),
                    4 => RetValue(Type::i32()),
                    8 => RetValue(Type::i64()),
                    _ => RetPointer
                }
            }
            _ => {
                RetPointer
            }
        };

        match strategy {
            RetValue(t) => {
                ret_ty = LLVMType {
                    cast: true,
                    ty: t
                };
                sret = false;
            }
            RetPointer => {
                arg_tys.push(LLVMType {
                    cast: false,
                    ty: rty.ptr_to()
                });
                attrs.push(Some(StructRetAttribute));

                ret_ty = LLVMType {
                    cast: false,
                    ty: Type::void(),
                };
                sret = true;
            }
        }
    } else {
        ret_ty = LLVMType {
            cast: false,
            ty: rty
        };
        sret = false;
    }

    for &a in atys.iter() {
        arg_tys.push(LLVMType { cast: false, ty: a });
        attrs.push(None);
    }

    return FnType {
        arg_tys: arg_tys,
        ret_ty: ret_ty,
        attrs: attrs,
        sret: sret
    };
}
