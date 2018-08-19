// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use abi::call::{FnType, ArgType, Uniform};
use abi::{HasDataLayout, LayoutOf, TyLayout, TyLayoutMethods};

// Data layout: e-p:32:32-i64:64-v128:32:128-n32-S128

// See the https://github.com/kripken/emscripten-fastcomp-clang repository.
// The class `EmscriptenABIInfo` in `/lib/CodeGen/TargetInfo.cpp` contains the ABI definitions.

fn classify_ret_ty<'a, Ty, C>(cx: C, ret: &mut ArgType<'a, Ty>)
    where Ty: TyLayoutMethods<'a, C> + Copy,
          C: LayoutOf<Ty = Ty, TyLayout = TyLayout<'a, Ty>> + HasDataLayout
{
    if ret.layout.is_aggregate() {
        if let Some(unit) = ret.layout.homogeneous_aggregate(cx) {
            let size = ret.layout.size;
            if unit.size == size {
                ret.cast_to(Uniform {
                    unit,
                    total: size
                });
                return;
            }
        }

        ret.make_indirect();
    }
}

fn classify_arg_ty<Ty>(arg: &mut ArgType<Ty>) {
    if arg.layout.is_aggregate() {
        arg.make_indirect_byval();
    }
}

pub fn compute_abi_info<'a, Ty, C>(cx: C, fty: &mut FnType<'a, Ty>)
    where Ty: TyLayoutMethods<'a, C> + Copy,
          C: LayoutOf<Ty = Ty, TyLayout = TyLayout<'a, Ty>> + HasDataLayout
{
    if !fty.ret.is_ignore() {
        classify_ret_ty(cx, &mut fty.ret);
    }

    for arg in &mut fty.args {
        if arg.is_ignore() { continue; }
        classify_arg_ty(arg);
    }
}
