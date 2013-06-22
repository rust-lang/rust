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
use super::cabi::*;
use super::common::*;
use super::machine::*;

use middle::trans::type_::Type;

struct X86_ABIInfo {
    ccx: @mut CrateContext
}

impl ABIInfo for X86_ABIInfo {
    fn compute_info(&self,
                    atys: &[Type],
                    rty: Type,
                    ret_def: bool) -> FnType {
        let mut arg_tys = do atys.map |a| {
            LLVMType { cast: false, ty: *a }
        };
        let mut ret_ty = LLVMType {
            cast: false,
            ty: rty
        };
        let mut attrs = do atys.map |_| {
            None
        };

        // Rules for returning structs taken from
        // http://www.angelcode.com/dev/callconv/callconv.html
        // Clang's ABI handling is in lib/CodeGen/TargetInfo.cpp
        let sret = {
            let returning_a_struct = rty.kind() == Struct && ret_def;
            let big_struct = match self.ccx.sess.targ_cfg.os {
                os_win32 | os_macos => llsize_of_alloc(self.ccx, rty) > 8,
                _ => true
            };
            returning_a_struct && big_struct
        };

        if sret {
            let ret_ptr_ty = LLVMType {
                cast: false,
                ty: ret_ty.ty.ptr_to()
            };
            arg_tys = ~[ret_ptr_ty] + arg_tys;
            attrs = ~[Some(StructRetAttribute)] + attrs;
            ret_ty = LLVMType {
                cast: false,
                ty: Type::void(),
            };
        } else if !ret_def {
            ret_ty = LLVMType {
                cast: false,
                ty: Type::void()
            };
        }

        return FnType {
            arg_tys: arg_tys,
            ret_ty: ret_ty,
            attrs: attrs,
            sret: sret
        };
    }
}

pub fn abi_info(ccx: @mut CrateContext) -> @ABIInfo {
    return @X86_ABIInfo {
        ccx: ccx
    } as @ABIInfo;
}
