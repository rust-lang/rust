// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// The classification code for the x86_64 ABI is taken from the clay language
// https://github.com/jckarter/clay/blob/master/compiler/src/externals.cpp

use abi::{ArgType, ArgAttribute, CastTarget, FnType, LayoutExt, Reg, RegKind};
use context::CrateContext;

use rustc::ty::layout::{self, Layout, FullLayout, Size};

#[derive(Clone, Copy, PartialEq, Debug)]
enum Class {
    None,
    Int,
    Sse,
    SseUp
}

#[derive(Clone, Copy, Debug)]
struct Memory;

// Currently supported vector size (AVX).
const LARGEST_VECTOR_SIZE: usize = 256;
const MAX_EIGHTBYTES: usize = LARGEST_VECTOR_SIZE / 64;

fn classify_arg<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, arg: &ArgType<'tcx>)
                          -> Result<[Class; MAX_EIGHTBYTES], Memory> {
    fn unify(cls: &mut [Class],
             off: Size,
             c: Class) {
        let i = (off.bytes() / 8) as usize;
        let to_write = match (cls[i], c) {
            (Class::None, _) => c,
            (_, Class::None) => return,

            (Class::Int, _) |
            (_, Class::Int) => Class::Int,

            (Class::Sse, _) |
            (_, Class::Sse) => Class::Sse,

            (Class::SseUp, Class::SseUp) => Class::SseUp
        };
        cls[i] = to_write;
    }

    fn classify<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                          layout: FullLayout<'tcx>,
                          cls: &mut [Class],
                          off: Size)
                          -> Result<(), Memory> {
        if !off.is_abi_aligned(layout.align(ccx)) {
            if layout.size(ccx).bytes() > 0 {
                return Err(Memory);
            }
            return Ok(());
        }

        match layout.abi {
            layout::Abi::Scalar(value) => {
                let reg = match value {
                    layout::Int(..) |
                    layout::Pointer => Class::Int,
                    layout::F32 |
                    layout::F64 => Class::Sse
                };
                unify(cls, off, reg);
            }

            layout::Abi::Vector { element, count } => {
                unify(cls, off, Class::Sse);

                // everything after the first one is the upper
                // half of a register.
                let eltsz = element.size(ccx);
                for i in 1..count {
                    unify(cls, off + eltsz * (i as u64), Class::SseUp);
                }
            }

            layout::Abi::Aggregate { .. } => {
                // FIXME(eddyb) have to work around Rust enums for now.
                // Fix is either guarantee no data where there is no field,
                // by putting variants in fields, or be more clever.
                match *layout.layout {
                    Layout::General { .. } |
                    Layout::NullablePointer { .. } => return Err(Memory),
                    _ => {}
                }
                for i in 0..layout.fields.count() {
                    let field_off = off + layout.fields.offset(i);
                    classify(ccx, layout.field(ccx, i), cls, field_off)?;
                }
            }

        }

        Ok(())
    }

    let n = ((arg.layout.size(ccx).bytes() + 7) / 8) as usize;
    if n > MAX_EIGHTBYTES {
        return Err(Memory);
    }

    let mut cls = [Class::None; MAX_EIGHTBYTES];
    classify(ccx, arg.layout, &mut cls, Size::from_bytes(0))?;
    if n > 2 {
        if cls[0] != Class::Sse {
            return Err(Memory);
        }
        if cls[1..n].iter().any(|&c| c != Class::SseUp) {
            return Err(Memory);
        }
    } else {
        let mut i = 0;
        while i < n {
            if cls[i] == Class::SseUp {
                cls[i] = Class::Sse;
            } else if cls[i] == Class::Sse {
                i += 1;
                while i != n && cls[i] == Class::SseUp { i += 1; }
            } else {
                i += 1;
            }
        }
    }

    Ok(cls)
}

fn reg_component(cls: &[Class], i: &mut usize, size: Size) -> Option<Reg> {
    if *i >= cls.len() {
        return None;
    }

    match cls[*i] {
        Class::None => None,
        Class::Int => {
            *i += 1;
            Some(match size.bytes() {
                1 => Reg::i8(),
                2 => Reg::i16(),
                3 |
                4 => Reg::i32(),
                _ => Reg::i64()
            })
        }
        Class::Sse => {
            let vec_len = 1 + cls[*i+1..].iter().take_while(|&&c| c == Class::SseUp).count();
            *i += vec_len;
            Some(if vec_len == 1 {
                match size.bytes() {
                    4 => Reg::f32(),
                    _ => Reg::f64()
                }
            } else {
                Reg {
                    kind: RegKind::Vector,
                    size: Size::from_bytes(8) * (vec_len as u64)
                }
            })
        }
        c => bug!("reg_component: unhandled class {:?}", c)
    }
}

fn cast_target(cls: &[Class], size: Size) -> CastTarget {
    let mut i = 0;
    let lo = reg_component(cls, &mut i, size).unwrap();
    let offset = Size::from_bytes(8) * (i as u64);
    let target = if size <= offset {
        CastTarget::from(lo)
    } else {
        let hi = reg_component(cls, &mut i, size - offset).unwrap();
        CastTarget::Pair(lo, hi)
    };
    assert_eq!(reg_component(cls, &mut i, Size::from_bytes(0)), None);
    target
}

pub fn compute_abi_info<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>, fty: &mut FnType<'tcx>) {
    let mut int_regs = 6; // RDI, RSI, RDX, RCX, R8, R9
    let mut sse_regs = 8; // XMM0-7

    let mut x86_64_ty = |arg: &mut ArgType<'tcx>, is_arg: bool| {
        let cls = classify_arg(ccx, arg);

        let mut needed_int = 0;
        let mut needed_sse = 0;
        let in_mem = match cls {
            Err(Memory) => true,
            Ok(ref cls) if is_arg => {
                for &c in cls {
                    match c {
                        Class::Int => needed_int += 1,
                        Class::Sse => needed_sse += 1,
                        _ => {}
                    }
                }
                arg.layout.is_aggregate() &&
                    (int_regs < needed_int || sse_regs < needed_sse)
            }
            Ok(_) => false
        };

        if in_mem {
            arg.make_indirect(ccx);
            if is_arg {
                arg.attrs.set(ArgAttribute::ByVal);
            } else {
                // `sret` parameter thus one less integer register available
                int_regs -= 1;
            }
        } else {
            // split into sized chunks passed individually
            int_regs -= needed_int;
            sse_regs -= needed_sse;

            if arg.layout.is_aggregate() {
                let size = arg.layout.size(ccx);
                arg.cast_to(cast_target(cls.as_ref().unwrap(), size))
            } else {
                arg.extend_integer_width_to(32);
            }
        }
    };

    if !fty.ret.is_ignore() {
        x86_64_ty(&mut fty.ret, false);
    }

    for arg in &mut fty.args {
        if arg.is_ignore() { continue; }
        x86_64_ty(arg, true);
    }
}
