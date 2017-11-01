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
use rustc::ty::layout;

#[derive(Debug, Clone, Copy, PartialEq)]
enum ABI {
    ELFv1, // original ABI used for powerpc64 (big-endian)
    ELFv2, // newer ABI used for powerpc64le
}
use self::ABI::*;

fn is_homogeneous_aggregate<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                      arg: &mut ArgType<'tcx>,
                                      abi: ABI)
                                     -> Option<Uniform> {
    arg.layout.homogeneous_aggregate(ccx).and_then(|unit| {
        let size = arg.layout.size(ccx);

        // ELFv1 only passes one-member aggregates transparently.
        // ELFv2 passes up to eight uniquely addressable members.
        if (abi == ELFv1 && size > unit.size)
                || size > unit.size.checked_mul(8, ccx).unwrap() {
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

fn classify_ret_ty<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, ret: &mut ArgType<'tcx>, abi: ABI) {
    if !ret.layout.is_aggregate() {
        ret.extend_integer_width_to(64);
        return;
    }

    // The ELFv1 ABI doesn't return aggregates in registers
    if abi == ELFv1 {
        ret.make_indirect(ccx);
        return;
    }

    if let Some(uniform) = is_homogeneous_aggregate(ccx, ret, abi) {
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

fn classify_arg_ty<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, arg: &mut ArgType<'tcx>, abi: ABI) {
    if !arg.layout.is_aggregate() {
        arg.extend_integer_width_to(64);
        return;
    }

    if let Some(uniform) = is_homogeneous_aggregate(ccx, arg, abi) {
        arg.cast_to(ccx, uniform);
        return;
    }

    let size = arg.layout.size(ccx);
    let (unit, total) = match abi {
        ELFv1 => {
            // In ELFv1, aggregates smaller than a doubleword should appear in
            // the least-significant bits of the parameter doubleword.  The rest
            // should be padded at their tail to fill out multiple doublewords.
            if size.bits() <= 64 {
                (Reg { kind: RegKind::Integer, size }, size)
            } else {
                let align = layout::Align::from_bits(64, 64).unwrap();
                (Reg::i64(), size.abi_align(align))
            }
        },
        ELFv2 => {
            // In ELFv2, we can just cast directly.
            (Reg::i64(), size)
        },
    };

    arg.cast_to(ccx, Uniform {
        unit,
        total
    });
}

pub fn compute_abi_info<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, fty: &mut FnType<'tcx>) {
    let abi = match ccx.sess().target.target.target_endian.as_str() {
        "big" => ELFv1,
        "little" => ELFv2,
        _ => unimplemented!(),
    };

    if !fty.ret.is_ignore() {
        classify_ret_ty(ccx, &mut fty.ret, abi);
    }

    for arg in &mut fty.args {
        if arg.is_ignore() { continue; }
        classify_arg_ty(ccx, arg, abi);
    }
}
