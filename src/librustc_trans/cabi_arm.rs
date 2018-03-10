// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use abi::{FnType, ArgType, LayoutExt, Reg, RegKind, Uniform};
use context::CodegenCx;
use llvm::CallConv;

fn is_homogeneous_aggregate<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>, arg: &mut ArgType<'tcx>)
                                     -> Option<Uniform> {
    arg.layout.homogeneous_aggregate(cx).and_then(|unit| {
        let size = arg.layout.size;

        // Ensure we have at most four uniquely addressable members.
        if size > unit.size.checked_mul(4, cx).unwrap() {
            return None;
        }

        let valid_unit = match unit.kind {
            RegKind::Integer => false,
            RegKind::Float => true,
            RegKind::Vector => size.bits() == 64 || size.bits() == 128
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

fn classify_ret_ty<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>, ret: &mut ArgType<'tcx>, vfp: bool) {
    if !ret.layout.is_aggregate() {
        ret.extend_integer_width_to(32);
        return;
    }

    if vfp {
        if let Some(uniform) = is_homogeneous_aggregate(cx, ret) {
            ret.cast_to(uniform);
            return;
        }
    }

    let size = ret.layout.size;
    let bits = size.bits();
    if bits <= 32 {
        let unit = if bits <= 8 {
            Reg::i8()
        } else if bits <= 16 {
            Reg::i16()
        } else {
            Reg::i32()
        };
        ret.cast_to(Uniform {
            unit,
            total: size
        });
        return;
    }
    ret.make_indirect();
}

fn classify_arg_ty<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>, arg: &mut ArgType<'tcx>, vfp: bool) {
    if !arg.layout.is_aggregate() {
        arg.extend_integer_width_to(32);
        return;
    }

    if vfp {
        if let Some(uniform) = is_homogeneous_aggregate(cx, arg) {
            arg.cast_to(uniform);
            return;
        }
    }

    let align = arg.layout.align.abi();
    let total = arg.layout.size;
    arg.cast_to(Uniform {
        unit: if align <= 4 { Reg::i32() } else { Reg::i64() },
        total
    });
}

pub fn compute_abi_info<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>, fty: &mut FnType<'tcx>) {
    // If this is a target with a hard-float ABI, and the function is not explicitly
    // `extern "aapcs"`, then we must use the VFP registers for homogeneous aggregates.
    let vfp = cx.sess().target.target.llvm_target.ends_with("hf")
        && fty.cconv != CallConv::ArmAapcsCallConv
        && !fty.variadic;

    if !fty.ret.is_ignore() {
        classify_ret_ty(cx, &mut fty.ret, vfp);
    }

    for arg in &mut fty.args {
        if arg.is_ignore() { continue; }
        classify_arg_ty(cx, arg, vfp);
    }
}
