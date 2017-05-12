// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use abi::{FnType, ArgType, LayoutExt, Reg, Uniform};
use context::CrateContext;

fn classify_ret_ty<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, ret: &mut ArgType<'tcx>) {
    if !ret.layout.is_aggregate() {
        ret.extend_integer_width_to(32);
        return;
    }
    let size = ret.layout.size(ccx);
    let bits = size.bits();
    if bits <= 32 {
        let unit = if bits <= 8 {
            Reg::i8()
        } else if bits <= 16 {
            Reg::i16()
        } else {
            Reg::i32()
        };
        ret.cast_to(ccx, Uniform {
            unit,
            total: size
        });
        return;
    }
    ret.make_indirect(ccx);
}

fn classify_arg_ty<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, arg: &mut ArgType<'tcx>) {
    if !arg.layout.is_aggregate() {
        arg.extend_integer_width_to(32);
        return;
    }
    let align = arg.layout.align(ccx).abi();
    let total = arg.layout.size(ccx);
    arg.cast_to(ccx, Uniform {
        unit: if align <= 4 { Reg::i32() } else { Reg::i64() },
        total
    });
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
