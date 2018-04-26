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

use abi::call::{FnType, ArgType, Reg, RegKind, Uniform};
use abi::{Align, Endian, HasDataLayout, LayoutOf, TyLayout, TyLayoutMethods};

#[derive(Debug, Clone, Copy, PartialEq)]
enum ABI {
    ELFv1, // original ABI used for powerpc64 (big-endian)
    ELFv2, // newer ABI used for powerpc64le
}
use self::ABI::*;

fn is_homogeneous_aggregate<'a, Ty, C>(cx: C, arg: &mut ArgType<'a, Ty>, abi: ABI)
                                       -> Option<Uniform>
    where Ty: TyLayoutMethods<'a, C> + Copy,
          C: LayoutOf<Ty = Ty, TyLayout = TyLayout<'a, Ty>> + HasDataLayout
{
    arg.layout.homogeneous_aggregate(cx).and_then(|unit| {
        // ELFv1 only passes one-member aggregates transparently.
        // ELFv2 passes up to eight uniquely addressable members.
        if (abi == ELFv1 && arg.layout.size > unit.size)
                || arg.layout.size > unit.size.checked_mul(8, cx).unwrap() {
            return None;
        }

        let valid_unit = match unit.kind {
            RegKind::Integer => false,
            RegKind::Float => true,
            RegKind::Vector => arg.layout.size.bits() == 128
        };

        if valid_unit {
            Some(Uniform {
                unit,
                total: arg.layout.size
            })
        } else {
            None
        }
    })
}

fn classify_ret_ty<'a, Ty, C>(cx: C, ret: &mut ArgType<'a, Ty>, abi: ABI)
    where Ty: TyLayoutMethods<'a, C> + Copy,
          C: LayoutOf<Ty = Ty, TyLayout = TyLayout<'a, Ty>> + HasDataLayout
{
    if !ret.layout.is_aggregate() {
        ret.extend_integer_width_to(64);
        return;
    }

    // The ELFv1 ABI doesn't return aggregates in registers
    if abi == ELFv1 {
        ret.make_indirect();
        return;
    }

    if let Some(uniform) = is_homogeneous_aggregate(cx, ret, abi) {
        ret.cast_to(uniform);
        return;
    }

    let size = ret.layout.size;
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

        ret.cast_to(Uniform {
            unit,
            total: size
        });
        return;
    }

    ret.make_indirect();
}

fn classify_arg_ty<'a, Ty, C>(cx: C, arg: &mut ArgType<'a, Ty>, abi: ABI)
    where Ty: TyLayoutMethods<'a, C> + Copy,
          C: LayoutOf<Ty = Ty, TyLayout = TyLayout<'a, Ty>> + HasDataLayout
{
    if !arg.layout.is_aggregate() {
        arg.extend_integer_width_to(64);
        return;
    }

    if let Some(uniform) = is_homogeneous_aggregate(cx, arg, abi) {
        arg.cast_to(uniform);
        return;
    }

    let size = arg.layout.size;
    let (unit, total) = match abi {
        ELFv1 => {
            // In ELFv1, aggregates smaller than a doubleword should appear in
            // the least-significant bits of the parameter doubleword.  The rest
            // should be padded at their tail to fill out multiple doublewords.
            if size.bits() <= 64 {
                (Reg { kind: RegKind::Integer, size }, size)
            } else {
                let align = Align::from_bits(64, 64).unwrap();
                (Reg::i64(), size.abi_align(align))
            }
        },
        ELFv2 => {
            // In ELFv2, we can just cast directly.
            (Reg::i64(), size)
        },
    };

    arg.cast_to(Uniform {
        unit,
        total
    });
}

pub fn compute_abi_info<'a, Ty, C>(cx: C, fty: &mut FnType<'a, Ty>)
    where Ty: TyLayoutMethods<'a, C> + Copy,
          C: LayoutOf<Ty = Ty, TyLayout = TyLayout<'a, Ty>> + HasDataLayout
{
    let abi = match cx.data_layout().endian {
        Endian::Big => ELFv1,
        Endian::Little => ELFv2,
    };

    if !fty.ret.is_ignore() {
        classify_ret_ty(cx, &mut fty.ret, abi);
    }

    for arg in &mut fty.args {
        if arg.is_ignore() { continue; }
        classify_arg_ty(cx, arg, abi);
    }
}
