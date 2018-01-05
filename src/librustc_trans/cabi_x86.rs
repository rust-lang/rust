// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use abi::{ArgAttribute, FnType, LayoutExt, PassMode, Reg, RegKind};
use common::CodegenCx;

use rustc::ty::layout::{self, TyLayout};

#[derive(PartialEq)]
pub enum Flavor {
    General,
    Fastcall
}

fn is_single_fp_element<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
                                  layout: TyLayout<'tcx>) -> bool {
    match layout.abi {
        layout::Abi::Scalar(ref scalar) => {
            match scalar.value {
                layout::F32 | layout::F64 => true,
                _ => false
            }
        }
        layout::Abi::Aggregate { .. } => {
            if layout.fields.count() == 1 && layout.fields.offset(0).bytes() == 0 {
                is_single_fp_element(cx, layout.field(cx, 0))
            } else {
                false
            }
        }
        _ => false
    }
}

pub fn compute_abi_info<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
                                  fty: &mut FnType<'tcx>,
                                  flavor: Flavor) {
    if !fty.ret.is_ignore() {
        if fty.ret.layout.is_aggregate() {
            // Returning a structure. Most often, this will use
            // a hidden first argument. On some platforms, though,
            // small structs are returned as integers.
            //
            // Some links:
            // http://www.angelcode.com/dev/callconv/callconv.html
            // Clang's ABI handling is in lib/CodeGen/TargetInfo.cpp
            let t = &cx.sess().target.target;
            if t.options.is_like_osx || t.options.is_like_windows
                || t.options.is_like_openbsd {
                // According to Clang, everyone but MSVC returns single-element
                // float aggregates directly in a floating-point register.
                if !t.options.is_like_msvc && is_single_fp_element(cx, fty.ret.layout) {
                    match fty.ret.layout.size.bytes() {
                        4 => fty.ret.cast_to(Reg::f32()),
                        8 => fty.ret.cast_to(Reg::f64()),
                        _ => fty.ret.make_indirect()
                    }
                } else {
                    match fty.ret.layout.size.bytes() {
                        1 => fty.ret.cast_to(Reg::i8()),
                        2 => fty.ret.cast_to(Reg::i16()),
                        4 => fty.ret.cast_to(Reg::i32()),
                        8 => fty.ret.cast_to(Reg::i64()),
                        _ => fty.ret.make_indirect()
                    }
                }
            } else {
                fty.ret.make_indirect();
            }
        } else {
            fty.ret.extend_integer_width_to(32);
        }
    }

    for arg in &mut fty.args {
        if arg.is_ignore() { continue; }
        if arg.layout.is_aggregate() {
            arg.make_indirect_byval();
        } else {
            arg.extend_integer_width_to(32);
        }
    }

    if flavor == Flavor::Fastcall {
        // Mark arguments as InReg like clang does it,
        // so our fastcall is compatible with C/C++ fastcall.

        // Clang reference: lib/CodeGen/TargetInfo.cpp
        // See X86_32ABIInfo::shouldPrimitiveUseInReg(), X86_32ABIInfo::updateFreeRegs()

        // IsSoftFloatABI is only set to true on ARM platforms,
        // which in turn can't be x86?

        let mut free_regs = 2;

        for arg in &mut fty.args {
            let attrs = match arg.mode {
                PassMode::Ignore |
                PassMode::Indirect(_) => continue,
                PassMode::Direct(ref mut attrs) => attrs,
                PassMode::Pair(..) |
                PassMode::Cast(_) => {
                    bug!("x86 shouldn't be passing arguments by {:?}", arg.mode)
                }
            };

            // At this point we know this must be a primitive of sorts.
            let unit = arg.layout.homogeneous_aggregate(cx).unwrap();
            assert_eq!(unit.size, arg.layout.size);
            if unit.kind == RegKind::Float {
                continue;
            }

            let size_in_regs = (arg.layout.size.bits() + 31) / 32;

            if size_in_regs == 0 {
                continue;
            }

            if size_in_regs > free_regs {
                break;
            }

            free_regs -= size_in_regs;

            if arg.layout.size.bits() <= 32 && unit.kind == RegKind::Integer {
                attrs.set(ArgAttribute::InReg);
            }

            if free_regs == 0 {
                break;
            }
        }
    }
}
