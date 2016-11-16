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

#![allow(non_upper_case_globals)]
use self::RegClass::*;

use llvm::{Integer, Pointer, Float, Double};
use llvm::{Struct, Array, Vector};
use abi::{self, ArgType, ArgAttribute, FnType};
use context::CrateContext;
use type_::Type;

#[derive(Clone, Copy, PartialEq)]
enum RegClass {
    NoClass,
    Int,
    SSEFs,
    SSEFv,
    SSEDs,
    SSEDv,
    SSEInt(/* bitwidth */ u64),
    /// Data that can appear in the upper half of an SSE register.
    SSEUp,
    X87,
    X87Up,
    ComplexX87,
    Memory
}

trait TypeMethods {
    fn is_reg_ty(&self) -> bool;
}

impl TypeMethods for Type {
    fn is_reg_ty(&self) -> bool {
        match self.kind() {
            Integer | Pointer | Float | Double => true,
            _ => false
        }
    }
}

impl RegClass {
    fn is_sse(&self) -> bool {
        match *self {
            SSEFs | SSEFv | SSEDs | SSEDv | SSEInt(_) => true,
            _ => false
        }
    }
}

trait ClassList {
    fn is_pass_byval(&self) -> bool;
    fn is_ret_bysret(&self) -> bool;
}

impl ClassList for [RegClass] {
    fn is_pass_byval(&self) -> bool {
        if self.is_empty() { return false; }

        let class = self[0];
           class == Memory
        || class == X87
        || class == ComplexX87
    }

    fn is_ret_bysret(&self) -> bool {
        if self.is_empty() { return false; }

        self[0] == Memory
    }
}

fn classify_ty(ty: Type) -> Vec<RegClass> {
    fn align(off: usize, ty: Type) -> usize {
        let a = ty_align(ty);
        return (off + a - 1) / a * a;
    }

    fn ty_align(ty: Type) -> usize {
        abi::ty_align(ty, 8)
    }

    fn ty_size(ty: Type) -> usize {
        abi::ty_size(ty, 8)
    }

    fn all_mem(cls: &mut [RegClass]) {
        for elt in cls {
            *elt = Memory;
        }
    }

    fn unify(cls: &mut [RegClass],
             i: usize,
             newv: RegClass) {
        if cls[i] == newv { return }

        let to_write = match (cls[i], newv) {
            (NoClass,     _) => newv,
            (_,           NoClass) => return,

            (Memory,      _) |
            (_,           Memory) => Memory,

            (Int,         _) |
            (_,           Int) => Int,

            (X87,         _) |
            (X87Up,       _) |
            (ComplexX87,  _) |
            (_,           X87) |
            (_,           X87Up) |
            (_,           ComplexX87) => Memory,

            (SSEFv,       SSEUp) |
            (SSEFs,       SSEUp) |
            (SSEDv,       SSEUp) |
            (SSEDs,       SSEUp) |
            (SSEInt(_),   SSEUp) => return,

            (..) => newv
        };
        cls[i] = to_write;
    }

    fn classify_struct(tys: &[Type],
                       cls: &mut [RegClass],
                       i: usize,
                       off: usize,
                       packed: bool) {
        let mut field_off = off;
        for ty in tys {
            if !packed {
                field_off = align(field_off, *ty);
            }
            classify(*ty, cls, i, field_off);
            field_off += ty_size(*ty);
        }
    }

    fn classify(ty: Type,
                cls: &mut [RegClass], ix: usize,
                off: usize) {
        let t_align = ty_align(ty);
        let t_size = ty_size(ty);

        let misalign = off % t_align;
        if misalign != 0 {
            let mut i = off / 8;
            let e = (off + t_size + 7) / 8;
            while i < e {
                unify(cls, ix + i, Memory);
                i += 1;
            }
            return;
        }

        match ty.kind() {
            Integer |
            Pointer => {
                unify(cls, ix + off / 8, Int);
            }
            Float => {
                if off % 8 == 4 {
                    unify(cls, ix + off / 8, SSEFv);
                } else {
                    unify(cls, ix + off / 8, SSEFs);
                }
            }
            Double => {
                unify(cls, ix + off / 8, SSEDs);
            }
            Struct => {
                classify_struct(&ty.field_types(), cls, ix, off, ty.is_packed());
            }
            Array => {
                let len = ty.array_length();
                let elt = ty.element_type();
                let eltsz = ty_size(elt);
                let mut i = 0;
                while i < len {
                    classify(elt, cls, ix, off + i * eltsz);
                    i += 1;
                }
            }
            Vector => {
                let len = ty.vector_length();
                let elt = ty.element_type();
                let eltsz = ty_size(elt);
                let mut reg = match elt.kind() {
                    Integer => SSEInt(elt.int_width()),
                    Float => SSEFv,
                    Double => SSEDv,
                    _ => bug!("classify: unhandled vector element type")
                };

                let mut i = 0;
                while i < len {
                    unify(cls, ix + (off + i * eltsz) / 8, reg);

                    // everything after the first one is the upper
                    // half of a register.
                    reg = SSEUp;
                    i += 1;
                }
            }
            _ => bug!("classify: unhandled type")
        }
    }

    fn fixup(ty: Type, cls: &mut [RegClass]) {
        let mut i = 0;
        let ty_kind = ty.kind();
        let e = cls.len();
        if cls.len() > 2 && (ty_kind == Struct || ty_kind == Array || ty_kind == Vector) {
            if cls[i].is_sse() {
                i += 1;
                while i < e {
                    if cls[i] != SSEUp {
                        all_mem(cls);
                        return;
                    }
                    i += 1;
                }
            } else {
                all_mem(cls);
                return
            }
        } else {
            while i < e {
                if cls[i] == Memory {
                    all_mem(cls);
                    return;
                }
                if cls[i] == X87Up {
                    // for darwin
                    // cls[i] = SSEDs;
                    all_mem(cls);
                    return;
                }
                if cls[i] == SSEUp {
                    cls[i] = SSEDv;
                } else if cls[i].is_sse() {
                    i += 1;
                    while i != e && cls[i] == SSEUp { i += 1; }
                } else if cls[i] == X87 {
                    i += 1;
                    while i != e && cls[i] == X87Up { i += 1; }
                } else {
                    i += 1;
                }
            }
        }
    }

    let words = (ty_size(ty) + 7) / 8;
    let mut cls = vec![NoClass; words];
    if words > 4 {
        all_mem(&mut cls);
        return cls;
    }
    classify(ty, &mut cls, 0, 0);
    fixup(ty, &mut cls);
    return cls;
}

fn llreg_ty(ccx: &CrateContext, cls: &[RegClass]) -> Type {
    fn llvec_len(cls: &[RegClass]) -> usize {
        let mut len = 1;
        for c in cls {
            if *c != SSEUp {
                break;
            }
            len += 1;
        }
        return len;
    }

    let mut tys = Vec::new();
    let mut i = 0;
    let e = cls.len();
    while i < e {
        match cls[i] {
            Int => {
                tys.push(Type::i64(ccx));
            }
            SSEFv | SSEDv | SSEInt(_) => {
                let (elts_per_word, elt_ty) = match cls[i] {
                    SSEFv => (2, Type::f32(ccx)),
                    SSEDv => (1, Type::f64(ccx)),
                    SSEInt(bits) => {
                        assert!(bits == 8 || bits == 16 || bits == 32 || bits == 64,
                                "llreg_ty: unsupported SSEInt width {}", bits);
                        (64 / bits, Type::ix(ccx, bits))
                    }
                    _ => bug!(),
                };
                let vec_len = llvec_len(&cls[i + 1..]);
                let vec_ty = Type::vector(&elt_ty, vec_len as u64 * elts_per_word);
                tys.push(vec_ty);
                i += vec_len;
                continue;
            }
            SSEFs => {
                tys.push(Type::f32(ccx));
            }
            SSEDs => {
                tys.push(Type::f64(ccx));
            }
            _ => bug!("llregtype: unhandled class")
        }
        i += 1;
    }
    if tys.len() == 1 && tys[0].kind() == Vector {
        // if the type contains only a vector, pass it as that vector.
        tys[0]
    } else {
        Type::struct_(ccx, &tys, false)
    }
}

pub fn compute_abi_info(ccx: &CrateContext, fty: &mut FnType) {
    fn x86_64_ty<F>(ccx: &CrateContext,
                    arg: &mut ArgType,
                    is_mem_cls: F,
                    ind_attr: Option<ArgAttribute>)
        where F: FnOnce(&[RegClass]) -> bool
    {
        if !arg.ty.is_reg_ty() {
            let cls = classify_ty(arg.ty);
            if is_mem_cls(&cls) {
                arg.make_indirect(ccx);
                if let Some(attr) = ind_attr {
                    arg.attrs.set(attr);
                }
            } else {
                arg.cast = Some(llreg_ty(ccx, &cls));
            }
        } else {
            arg.extend_integer_width_to(32);
        }
    }

    let mut int_regs = 6; // RDI, RSI, RDX, RCX, R8, R9
    let mut sse_regs = 8; // XMM0-7

    if !fty.ret.is_ignore() {
        x86_64_ty(ccx, &mut fty.ret, |cls| {
            if cls.is_ret_bysret() {
                // `sret` parameter thus one less register available
                int_regs -= 1;
                true
            } else {
                false
            }
        }, None);
    }

    for arg in &mut fty.args {
        if arg.is_ignore() { continue; }
        x86_64_ty(ccx, arg, |cls| {
            let needed_int = cls.iter().filter(|&&c| c == Int).count() as isize;
            let needed_sse = cls.iter().filter(|c| c.is_sse()).count() as isize;
            let in_mem = cls.is_pass_byval() ||
                         int_regs < needed_int ||
                         sse_regs < needed_sse;
            if in_mem {
                // `byval` parameter thus one less integer register available
                int_regs -= 1;
            } else {
                // split into sized chunks passed individually
                int_regs -= needed_int;
                sse_regs -= needed_sse;
            }
            in_mem
        }, Some(ArgAttribute::ByVal));

        // An integer, pointer, double or float parameter
        // thus the above closure passed to `x86_64_ty` won't
        // get called.
        match arg.ty.kind() {
            Integer | Pointer => int_regs -= 1,
            Double | Float => sse_regs -= 1,
            _ => {}
        }
    }
}
