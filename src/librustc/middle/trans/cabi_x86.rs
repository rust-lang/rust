// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use driver::session::{os_win32, os_macos};
use lib::llvm::*;
use lib::llvm::llvm::*;
use super::cabi::*;
use super::common::*;
use super::machine::*;

struct X86_ABIInfo {
    ccx: @CrateContext
}

impl ABIInfo for X86_ABIInfo {
    fn compute_info(&self,
                    atys: &[TypeRef],
                    rty: TypeRef,
                    ret_def: bool) -> FnType {
        let mut arg_tys = ~[];
        let mut attrs = ~[];

        let ret_ty;
        let sret;
        if !ret_def {
            ret_ty = LLVMType {
                cast: false,
                ty: T_void(),
            };
            sret = false;
        } else if unsafe { LLVMGetTypeKind(rty) == Struct } {
            // Returning a structure. Most often, this will use
            // a hidden first argument. On some platforms, though,
            // small structs are returned as integers.
            //
            // Some links:
            // http://www.angelcode.com/dev/callconv/callconv.html
            // Clang's ABI handling is in lib/CodeGen/TargetInfo.cpp

            enum Strategy { RetValue(TypeRef), RetPointer }
            let strategy = match self.ccx.sess.targ_cfg.os {
                os_win32 | os_macos => {
                    match llsize_of_alloc(self.ccx, rty) {
                        1 => RetValue(T_i8()),
                        2 => RetValue(T_i16()),
                        4 => RetValue(T_i32()),
                        8 => RetValue(T_i64()),
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
                        ty: T_ptr(rty)
                    });
                    attrs.push(Some(StructRetAttribute));

                    ret_ty = LLVMType {
                        cast: false,
                        ty: T_void(),
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

        for atys.each |&a| {
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
}

pub fn abi_info(ccx: @CrateContext) -> @ABIInfo {
    return @X86_ABIInfo {
        ccx: ccx
    } as @ABIInfo;
}
