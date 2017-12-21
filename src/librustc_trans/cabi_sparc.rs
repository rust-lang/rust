// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cmp;
use abi::{align_up_to, ArgType, FnType, LayoutExt, Reg, Uniform};
use context::CrateContext;

fn classify_ret_ty<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, ret: &mut ArgType<'tcx>) {
    if !ret.layout.is_aggregate() {
        ret.extend_integer_width_to(32);
    } else {
        ret.make_indirect(ccx);
    }
}

fn classify_arg_ty(ccx: &CrateContext, arg: &mut ArgType, offset: &mut u64) {
    let size = arg.layout.size(ccx);
    let mut align = arg.layout.align(ccx).abi();
    align = cmp::min(cmp::max(align, 4), 8);

    if arg.layout.is_aggregate() {
        arg.cast_to(ccx, Uniform {
            unit: Reg::i32(),
            total: size
        });
        if ((align - 1) & *offset) > 0 {
            arg.pad_with(ccx, Reg::i32());
        }
    } else {
        arg.extend_integer_width_to(32)
    }

    *offset = align_up_to(*offset, align);
    *offset += align_up_to(size.bytes(), align);
}

pub fn compute_abi_info<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, fty: &mut FnType<'tcx>) {
    if !fty.ret.is_ignore() {
        classify_ret_ty(ccx, &mut fty.ret);
    }

    let mut offset = if fty.ret.is_indirect() { 4 } else { 0 };
    for arg in &mut fty.args {
        if arg.is_ignore() { continue; }
        classify_arg_ty(ccx, arg, &mut offset);
    }
}
