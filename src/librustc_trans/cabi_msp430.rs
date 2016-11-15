// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Reference: MSP430 Embedded Application Binary Interface
// http://www.ti.com/lit/an/slaa534/slaa534.pdf

#![allow(non_upper_case_globals)]

use llvm::Struct;

use abi::{self, ArgType, FnType};
use context::CrateContext;
use type_::Type;

fn ty_size(ty: Type) -> usize {
    abi::ty_size(ty, 2)
}

// 3.5 Structures or Unions Passed and Returned by Reference
//
// "Structures (including classes) and unions larger than 32 bits are passed and
// returned by reference. To pass a structure or union by reference, the caller
// places its address in the appropriate location: either in a register or on
// the stack, according to its position in the argument list. (..)"
fn classify_ret_ty(ccx: &CrateContext, ret: &mut ArgType) {
    if ret.ty.kind() == Struct && ty_size(ret.ty) > 32 {
        ret.make_indirect(ccx);
    } else {
        ret.extend_integer_width_to(16);
    }
}

fn classify_arg_ty(ccx: &CrateContext, arg: &mut ArgType) {
    if arg.ty.kind() == Struct && ty_size(arg.ty) > 32 {
        arg.make_indirect(ccx);
    } else {
        arg.extend_integer_width_to(16);
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
