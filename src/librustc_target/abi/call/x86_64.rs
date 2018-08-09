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

use abi::call::{ArgType, CastTarget, FnType, Reg, RegKind};
use abi::{self, Abi, HasDataLayout, LayoutOf, Size, TyLayout, TyLayoutMethods};

/// Classification of "eightbyte" components.
// NB: the order of the variants is from general to specific,
// such that `unify(a, b)` is the "smaller" of `a` and `b`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Class {
    Int,
    Sse,
    SseUp
}

#[derive(Clone, Copy, Debug)]
struct Memory;

// Currently supported vector size (AVX-512).
const LARGEST_VECTOR_SIZE: usize = 512;
const MAX_EIGHTBYTES: usize = LARGEST_VECTOR_SIZE / 64;

fn classify_arg<'a, Ty, C>(cx: C, arg: &ArgType<'a, Ty>)
                          -> Result<[Option<Class>; MAX_EIGHTBYTES], Memory>
    where Ty: TyLayoutMethods<'a, C> + Copy,
          C: LayoutOf<Ty = Ty, TyLayout = TyLayout<'a, Ty>> + HasDataLayout
{
    fn classify<'a, Ty, C>(cx: C, layout: TyLayout<'a, Ty>,
                          cls: &mut [Option<Class>], off: Size) -> Result<(), Memory>
        where Ty: TyLayoutMethods<'a, C> + Copy,
            C: LayoutOf<Ty = Ty, TyLayout = TyLayout<'a, Ty>> + HasDataLayout
    {
        if !off.is_abi_aligned(layout.align) {
            if !layout.is_zst() {
                return Err(Memory);
            }
            return Ok(());
        }

        let mut c = match layout.abi {
            Abi::Uninhabited => return Ok(()),

            Abi::Scalar(ref scalar) => {
                match scalar.value {
                    abi::Int(..) |
                    abi::Pointer => Class::Int,
                    abi::Float(_) => Class::Sse
                }
            }

            Abi::Vector { .. } => Class::Sse,

            Abi::ScalarPair(..) |
            Abi::Aggregate { .. } => {
                match layout.variants {
                    abi::Variants::Single { .. } => {
                        for i in 0..layout.fields.count() {
                            let field_off = off + layout.fields.offset(i);
                            classify(cx, layout.field(cx, i), cls, field_off)?;
                        }
                        return Ok(());
                    }
                    abi::Variants::Tagged { .. } |
                    abi::Variants::NicheFilling { .. } => return Err(Memory),
                }
            }

        };

        // Fill in `cls` for scalars (Int/Sse) and vectors (Sse).
        let first = (off.bytes() / 8) as usize;
        let last = ((off.bytes() + layout.size.bytes() - 1) / 8) as usize;
        for cls in &mut cls[first..=last] {
            *cls = Some(cls.map_or(c, |old| old.min(c)));

            // Everything after the first Sse "eightbyte"
            // component is the upper half of a register.
            if c == Class::Sse {
                c = Class::SseUp;
            }
        }

        Ok(())
    }

    let n = ((arg.layout.size.bytes() + 7) / 8) as usize;
    if n > MAX_EIGHTBYTES {
        return Err(Memory);
    }

    let mut cls = [None; MAX_EIGHTBYTES];
    classify(cx, arg.layout, &mut cls, Size::ZERO)?;
    if n > 2 {
        if cls[0] != Some(Class::Sse) {
            return Err(Memory);
        }
        if cls[1..n].iter().any(|&c| c != Some(Class::SseUp)) {
            return Err(Memory);
        }
    } else {
        let mut i = 0;
        while i < n {
            if cls[i] == Some(Class::SseUp) {
                cls[i] = Some(Class::Sse);
            } else if cls[i] == Some(Class::Sse) {
                i += 1;
                while i != n && cls[i] == Some(Class::SseUp) { i += 1; }
            } else {
                i += 1;
            }
        }
    }

    Ok(cls)
}

fn reg_component(cls: &[Option<Class>], i: &mut usize, size: Size) -> Option<Reg> {
    if *i >= cls.len() {
        return None;
    }

    match cls[*i] {
        None => None,
        Some(Class::Int) => {
            *i += 1;
            Some(if size.bytes() < 8 {
                Reg {
                    kind: RegKind::Integer,
                    size
                }
            } else {
                Reg::i64()
            })
        }
        Some(Class::Sse) => {
            let vec_len = 1 + cls[*i+1..].iter()
                .take_while(|&&c| c == Some(Class::SseUp))
                .count();
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
        Some(c) => unreachable!("reg_component: unhandled class {:?}", c)
    }
}

fn cast_target(cls: &[Option<Class>], size: Size) -> CastTarget {
    let mut i = 0;
    let lo = reg_component(cls, &mut i, size).unwrap();
    let offset = Size::from_bytes(8) * (i as u64);
    let mut target = CastTarget::from(lo);
    if size > offset {
        if let Some(hi) = reg_component(cls, &mut i, size - offset) {
            target = CastTarget::pair(lo, hi);
        }
    }
    assert_eq!(reg_component(cls, &mut i, Size::ZERO), None);
    target
}

pub fn compute_abi_info<'a, Ty, C>(cx: C, fty: &mut FnType<'a, Ty>)
    where Ty: TyLayoutMethods<'a, C> + Copy,
          C: LayoutOf<Ty = Ty, TyLayout = TyLayout<'a, Ty>> + HasDataLayout
{
    let mut int_regs = 6; // RDI, RSI, RDX, RCX, R8, R9
    let mut sse_regs = 8; // XMM0-7

    let mut x86_64_ty = |arg: &mut ArgType<'a, Ty>, is_arg: bool| {
        let mut cls_or_mem = classify_arg(cx, arg);

        let mut needed_int = 0;
        let mut needed_sse = 0;
        if is_arg {
            if let Ok(cls) = cls_or_mem {
                for &c in &cls {
                    match c {
                        Some(Class::Int) => needed_int += 1,
                        Some(Class::Sse) => needed_sse += 1,
                        _ => {}
                    }
                }
                if arg.layout.is_aggregate() && (int_regs < needed_int || sse_regs < needed_sse) {
                    cls_or_mem = Err(Memory);
                }
            }
        }

        match cls_or_mem {
            Err(Memory) => {
                if is_arg {
                    arg.make_indirect_byval();
                } else {
                    // `sret` parameter thus one less integer register available
                    arg.make_indirect();
                    int_regs -= 1;
                }
            }
            Ok(ref cls) => {
                // split into sized chunks passed individually
                int_regs -= needed_int;
                sse_regs -= needed_sse;

                if arg.layout.is_aggregate() {
                    let size = arg.layout.size;
                    arg.cast_to(cast_target(cls, size))
                } else {
                    arg.extend_integer_width_to(32);
                }
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
