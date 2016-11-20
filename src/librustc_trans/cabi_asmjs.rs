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

use llvm::{Struct, Array};
use abi::{FnType, ArgType, ArgAttribute};
use context::CrateContext;

// Data layout: e-p:32:32-i64:64-v128:32:128-n32-S128

// See the https://github.com/kripken/emscripten-fastcomp-clang repository.
// The class `EmscriptenABIInfo` in `/lib/CodeGen/TargetInfo.cpp` contains the ABI definitions.

fn classify_ret_ty(ccx: &CrateContext, ret: &mut ArgType) {
    match ret.ty.kind() {
        Struct => {
            let field_types = ret.ty.field_types();
            if field_types.len() == 1 {
                ret.cast = Some(field_types[0]);
            } else {
                ret.make_indirect(ccx);
            }
        }
        Array => {
            ret.make_indirect(ccx);
        }
        _ => {}
    }
}

fn classify_arg_ty(ccx: &CrateContext, arg: &mut ArgType) {
    if arg.ty.is_aggregate() {
        arg.make_indirect(ccx);
        arg.attrs.set(ArgAttribute::ByVal);
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
