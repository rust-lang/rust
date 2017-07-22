// Copyright 2014-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME:
// Alignment of 128 bit types is not currently handled, this will
// need to be fixed when PowerPC vector support is added.

use abi::{FnType, ArgType, LayoutExt, Reg, RegKind, Uniform};
use context::CrateContext;

fn is_homogeneous_aggregate<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, arg: &mut ArgType<'tcx>)
                                     -> Option<Uniform> {
    arg.layout.homogeneous_aggregate(ccx).and_then(|unit| {
        let size = arg.layout.size(ccx);

        // Ensure we have at most eight uniquely addressable members.
        if size > unit.size.checked_mul(8, ccx).unwrap() {
            return None;
        }

        let valid_unit = match unit.kind {
            RegKind::Integer => false,
            RegKind::Float => true,
            RegKind::Vector => size.bits() == 128
        };

        if valid_unit {
            Some(Uniform {
                unit,
                total: size
            })
        } else {
            None
        }
    })
}

fn classify_ret_ty<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, ret: &mut ArgType<'tcx>) {
    if !ret.layout.is_aggregate() {
        ret.extend_integer_width_to(64);
        return;
    }

    // The PowerPC64 big endian ABI doesn't return aggregates in registers
    if ccx.sess().target.target.target_endian == "big" {
        ret.make_indirect(ccx);
    }

    if let Some(uniform) = is_homogeneous_aggregate(ccx, ret) {
        ret.cast_to(ccx, uniform);
        return;
    }
    let size = ret.layout.size(ccx);
    let bits = size.bits();
    if bits <= 128 {
        let unit = if bits <= 8 {
            Reg::i8()
        } else if bits <= 16 {
            Reg::i16()
        } else if bits <= 32 {
            Reg::i32()
        } else {
            Reg::i64()
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
        arg.extend_integer_width_to(64);
        return;
    }

    if let Some(uniform) = is_homogeneous_aggregate(ccx, arg) {
        arg.cast_to(ccx, uniform);
        return;
    }

    let total = arg.layout.size(ccx);
    arg.cast_to(ccx, Uniform {
        unit: Reg::i64(),
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
