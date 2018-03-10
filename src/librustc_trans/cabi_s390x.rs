// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME: The assumes we're using the non-vector ABI, i.e. compiling
// for a pre-z13 machine or using -mno-vx.

use abi::{FnType, ArgType, LayoutExt, Reg};
use context::CodegenCx;

use rustc::ty::layout::{self, TyLayout};

fn classify_ret_ty(ret: &mut ArgType) {
    if !ret.layout.is_aggregate() && ret.layout.size.bits() <= 64 {
        ret.extend_integer_width_to(64);
    } else {
        ret.make_indirect();
    }
}

fn is_single_fp_element<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
                                  layout: TyLayout<'tcx>) -> bool {
    match layout.abi {
        layout::Abi::Scalar(ref scalar) => {
            match scalar.value {
                layout::F32 | layout::F64 => true,
                _ => false
            }
        }
        layout::Abi::Aggregate { .. } => {
            if layout.fields.count() == 1 && layout.fields.offset(0).bytes() == 0 {
                is_single_fp_element(cx, layout.field(cx, 0))
            } else {
                false
            }
        }
        _ => false
    }
}

fn classify_arg_ty<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>, arg: &mut ArgType<'tcx>) {
    if !arg.layout.is_aggregate() && arg.layout.size.bits() <= 64 {
        arg.extend_integer_width_to(64);
        return;
    }

    if is_single_fp_element(cx, arg.layout) {
        match arg.layout.size.bytes() {
            4 => arg.cast_to(Reg::f32()),
            8 => arg.cast_to(Reg::f64()),
            _ => arg.make_indirect()
        }
    } else {
        match arg.layout.size.bytes() {
            1 => arg.cast_to(Reg::i8()),
            2 => arg.cast_to(Reg::i16()),
            4 => arg.cast_to(Reg::i32()),
            8 => arg.cast_to(Reg::i64()),
            _ => arg.make_indirect()
        }
    }
}

pub fn compute_abi_info<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>, fty: &mut FnType<'tcx>) {
    if !fty.ret.is_ignore() {
        classify_ret_ty(&mut fty.ret);
    }

    for arg in &mut fty.args {
        if arg.is_ignore() { continue; }
        classify_arg_ty(cx, arg);
    }
}
