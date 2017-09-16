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
use context::CrateContext;

use rustc::ty::layout::{self, FullLayout};

fn classify_ret_ty<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, ret: &mut ArgType<'tcx>) {
    if !ret.layout.is_aggregate() && ret.layout.size(ccx).bits() <= 64 {
        ret.extend_integer_width_to(64);
    } else {
        ret.make_indirect(ccx);
    }
}

fn is_single_fp_element<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                  layout: FullLayout<'tcx>) -> bool {
    match layout.abi {
        layout::Abi::Scalar(layout::F32) |
        layout::Abi::Scalar(layout::F64) => true,
        layout::Abi::Aggregate => {
            if layout.fields.count() == 1 && layout.fields.offset(0).bytes() == 0 {
                is_single_fp_element(ccx, layout.field(ccx, 0))
            } else {
                false
            }
        }
        _ => false
    }
}

fn classify_arg_ty<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, arg: &mut ArgType<'tcx>) {
    let size = arg.layout.size(ccx);
    if !arg.layout.is_aggregate() && size.bits() <= 64 {
        arg.extend_integer_width_to(64);
        return;
    }

    if is_single_fp_element(ccx, arg.layout) {
        match size.bytes() {
            4 => arg.cast_to(Reg::f32()),
            8 => arg.cast_to(Reg::f64()),
            _ => arg.make_indirect(ccx)
        }
    } else {
        match size.bytes() {
            1 => arg.cast_to(Reg::i8()),
            2 => arg.cast_to(Reg::i16()),
            4 => arg.cast_to(Reg::i32()),
            8 => arg.cast_to(Reg::i64()),
            _ => arg.make_indirect(ccx)
        }
    }
}

pub fn compute_abi_info<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, fty: &mut FnType<'tcx>) {
    if !fty.ret.is_ignore() {
        classify_ret_ty(ccx, &mut fty.ret);
    }

    for arg in &mut fty.args {
        if arg.is_ignore() { continue; }
        classify_arg_ty(ccx, arg);
    }
}
