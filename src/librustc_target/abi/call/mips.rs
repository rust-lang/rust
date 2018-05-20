// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use abi::call::{ArgType, FnType, Reg, Uniform};
use abi::{HasDataLayout, LayoutOf, Size, TyLayoutMethods};

fn classify_ret_ty<'a, Ty, C>(cx: C, ret: &mut ArgType<Ty>, offset: &mut Size)
    where Ty: TyLayoutMethods<'a, C>, C: LayoutOf<Ty = Ty> + HasDataLayout
{
    if !ret.layout.is_aggregate() {
        ret.extend_integer_width_to(32);
    } else {
        ret.make_indirect();
        *offset += cx.data_layout().pointer_size;
    }
}

fn classify_arg_ty<'a, Ty, C>(cx: C, arg: &mut ArgType<Ty>, offset: &mut Size)
    where Ty: TyLayoutMethods<'a, C>, C: LayoutOf<Ty = Ty> + HasDataLayout
{
    let dl = cx.data_layout();
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

pub fn compute_abi_info<'a, Ty, C>(cx: C, fty: &mut FnType<Ty>)
    where Ty: TyLayoutMethods<'a, C>, C: LayoutOf<Ty = Ty> + HasDataLayout
{
    let mut offset = Size::ZERO;
    if !fty.ret.is_ignore() {
        classify_ret_ty(cx, &mut fty.ret, &mut offset);
    }

    for arg in &mut fty.args {
        if arg.is_ignore() { continue; }
        classify_arg_ty(cx, arg, &mut offset);
    }
}
