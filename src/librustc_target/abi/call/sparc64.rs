// Copyright 2014-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME: This needs an audit for correctness and completeness.

use abi::call::{FnType, ArgType, Reg, RegKind, Uniform};
use abi::{HasDataLayout, LayoutOf, TyLayout, TyLayoutMethods};

fn is_homogeneous_aggregate<'a, Ty, C>(cx: C, arg: &mut ArgType<'a, Ty>)
                                     -> Option<Uniform>
    where Ty: TyLayoutMethods<'a, C> + Copy,
          C: LayoutOf<Ty = Ty, TyLayout = TyLayout<'a, Ty>> + HasDataLayout
{
    arg.layout.homogeneous_aggregate(cx).and_then(|unit| {
        // Ensure we have at most eight uniquely addressable members.
        if arg.layout.size > unit.size.checked_mul(8, cx).unwrap() {
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

fn classify_ret_ty<'a, Ty, C>(cx: C, ret: &mut ArgType<'a, Ty>)
    where Ty: TyLayoutMethods<'a, C> + Copy,
          C: LayoutOf<Ty = Ty, TyLayout = TyLayout<'a, Ty>> + HasDataLayout
{
    if !ret.layout.is_aggregate() {
        ret.extend_integer_width_to(64);
        return;
    }

    if let Some(uniform) = is_homogeneous_aggregate(cx, ret) {
        ret.cast_to(uniform);
        return;
    }
    let size = ret.layout.size;
    let bits = size.bits();
    if bits <= 256 {
        let unit = Reg::i64();
        ret.cast_to(Uniform {
            unit,
            total: size
        });
        return;
    }

    // don't return aggregates in registers
    ret.make_indirect();
}

fn classify_arg_ty<'a, Ty, C>(cx: C, arg: &mut ArgType<'a, Ty>)
    where Ty: TyLayoutMethods<'a, C> + Copy,
          C: LayoutOf<Ty = Ty, TyLayout = TyLayout<'a, Ty>> + HasDataLayout
{
    if !arg.layout.is_aggregate() {
        arg.extend_integer_width_to(64);
        return;
    }

    if let Some(uniform) = is_homogeneous_aggregate(cx, arg) {
        arg.cast_to(uniform);
        return;
    }

    let total = arg.layout.size;
    if total.bits() > 128 {
        arg.make_indirect();
        return;
    }

    arg.cast_to(Uniform {
        unit: Reg::i64(),
        total
    });
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
        classify_arg_ty(cx, arg);
    }
}
