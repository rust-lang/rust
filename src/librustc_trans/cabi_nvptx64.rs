// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Reference: PTX Writer's Guide to Interoperability
// http://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability

#![allow(non_upper_case_globals)]

use llvm::Struct;

use abi::{self, ArgType, FnType};
use context::CrateContext;
use type_::Type;

fn ty_size(ty: Type) -> usize {
    abi::ty_size(ty, 8)
}

fn classify_ret_ty(ccx: &CrateContext, ret: &mut ArgType) {
    if ret.ty.kind() == Struct && ty_size(ret.ty) > 64 {
        ret.make_indirect(ccx);
    } else {
        ret.extend_integer_width_to(64);
    }
}

fn classify_arg_ty(ccx: &CrateContext, arg: &mut ArgType) {
    if arg.ty.kind() == Struct && ty_size(arg.ty) > 64 {
        arg.make_indirect(ccx);
    } else {
        arg.extend_integer_width_to(64);
    }
}

pub fn compute_abi_info(ccx: &CrateContext, fty: &mut FnType) {
    if !fty.ret.is_ignore() {
        classify_ret_ty(ccx, &mut fty.ret);
    }

    for arg in &mut fty.args {
        if arg.is_ignore() {
            continue;
        }
        classify_arg_ty(ccx, arg);
    }
}
