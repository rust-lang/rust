// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use abi::{ArgType, FnType, LayoutExt, Reg, Uniform};
use context::CrateContext;

use rustc::ty::layout::Size;

fn classify_ret_ty<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                             ret: &mut ArgType<'tcx>,
                             offset: &mut Size) {
    if !ret.layout.is_aggregate() {
        ret.extend_integer_width_to(32);
    } else {
        ret.make_indirect();
        *offset += ccx.tcx().data_layout.pointer_size;
    }
}

fn classify_arg_ty(ccx: &CrateContext, arg: &mut ArgType, offset: &mut Size) {
    let dl = &ccx.tcx().data_layout;
    let size = arg.layout.size;
    let align = arg.layout.align.max(dl.i32_align).min(dl.i64_align);

    if arg.layout.is_aggregate() {
        arg.cast_to(Uniform {
            unit: Reg::i32(),
            total: size
        });
        if !offset.is_abi_aligned(align) {
            arg.pad_with(Reg::i32());
        }
    } else {
        arg.extend_integer_width_to(32);
    }

    *offset = offset.abi_align(align) + size.abi_align(align);
}

pub fn compute_abi_info<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, fty: &mut FnType<'tcx>) {
    let mut offset = Size::from_bytes(0);
    if !fty.ret.is_ignore() {
        classify_ret_ty(ccx, &mut fty.ret, &mut offset);
    }

    for arg in &mut fty.args {
        if arg.is_ignore() { continue; }
        classify_arg_ty(ccx, arg, &mut offset);
    }
}
