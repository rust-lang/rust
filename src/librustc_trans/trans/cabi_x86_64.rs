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
use llvm::{Struct, Array, Attribute, Vector};
use llvm::{StructRetAttribute, ByValAttribute, ZExtAttribute};
use trans::cabi::{ArgType, FnType};
use trans::context::CrateContext;
use trans::type_::Type;

use std::cmp;
use std::iter::repeat;

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
        if self.len() == 0 { return false; }

        let class = self[0];
           class == Memory
        || class == X87
        || class == ComplexX87
    }

    fn is_ret_bysret(&self) -> bool {
        if self.len() == 0 { return false; }

        self[0] == Memory
    }
}

fn classify_ty(ty: Type) -> Vec<RegClass> {
    fn align(off: uint, ty: Type) -> uint {
        let a = ty_align(ty);
        return (off + a - 1u) / a * a;
    }

    fn ty_align(ty: Type) -> uint {
        match ty.kind() {
            Integer => ((ty.int_width() as uint) + 7) / 8,
            Pointer => 8,
            Float => 4,
            Double => 8,
            Struct => {
              if ty.is_packed() {
                1
              } else {
                let str_tys = ty.field_types();
                str_tys.iter().fold(1, |a, t| cmp::max(a, ty_align(*t)))
              }
            }
            Array => {
                let elt = ty.element_type();
                ty_align(elt)
            }
            Vector => {
                let len = ty.vector_length();
                let elt = ty.element_type();
                ty_align(elt) * len
            }
            _ => panic!("ty_align: unhandled type")
        }
    }

    fn ty_size(ty: Type) -> uint {
        match ty.kind() {
            Integer => (ty.int_width() as uint + 7) / 8,
            Pointer => 8,
            Float => 4,
            Double => 8,
            Struct => {
                let str_tys = ty.field_types();
                if ty.is_packed() {
                    str_tys.iter().fold(0, |s, t| s + ty_size(*t))
                } else {
                    let size = str_tys.iter().fold(0, |s, t| align(s, *t) + ty_size(*t));
                    align(size, ty)
                }
            }
            Array => {
                let len = ty.array_length();
                let elt = ty.element_type();
                let eltsz = ty_size(elt);
                len * eltsz
            }
            Vector => {
                let len = ty.vector_length();
                let elt = ty.element_type();
                let eltsz = ty_size(elt);
                len * eltsz
            }

            _ => panic!("ty_size: unhandled type")
        }
    }

    fn all_mem(cls: &mut [RegClass]) {
        for elt in cls.iter_mut() {
            *elt = Memory;
        }
    }

    fn unify(cls: &mut [RegClass],
             i: uint,
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

            (_,           _) => newv
        };
        cls[i] = to_write;
    }

    fn classify_struct(tys: &[Type],
                       cls: &mut [RegClass],
                       i: uint,
                       off: uint,
                       packed: bool) {
        let mut field_off = off;
        for ty in tys.iter() {
            if !packed {
                field_off = align(field_off, *ty);
            }
            classify(*ty, cls, i, field_off);
            field_off += ty_size(*ty);
        }
    }

    fn classify(ty: Type,
                cls: &mut [RegClass], ix: uint,
                off: uint) {
        let t_align = ty_align(ty);
        let t_size = ty_size(ty);

        let misalign = off % t_align;
        if misalign != 0u {
            let mut i = off / 8u;
            let e = (off + t_size + 7u) / 8u;
            while i < e {
                unify(cls, ix + i, Memory);
                i += 1u;
            }
            return;
        }

        match ty.kind() {
            Integer |
            Pointer => {
                unify(cls, ix + off / 8u, Int);
            }
            Float => {
                if off % 8u == 4u {
                    unify(cls, ix + off / 8u, SSEFv);
                } else {
                    unify(cls, ix + off / 8u, SSEFs);
                }
            }
            Double => {
                unify(cls, ix + off / 8u, SSEDs);
            }
            Struct => {
                classify_struct(ty.field_types().as_slice(), cls, ix, off, ty.is_packed());
            }
            Array => {
                let len = ty.array_length();
                let elt = ty.element_type();
                let eltsz = ty_size(elt);
                let mut i = 0u;
                while i < len {
                    classify(elt, cls, ix, off + i * eltsz);
                    i += 1u;
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
                    _ => panic!("classify: unhandled vector element type")
                };

                let mut i = 0u;
                while i < len {
                    unify(cls, ix + (off + i * eltsz) / 8, reg);

                    // everything after the first one is the upper
                    // half of a register.
                    reg = SSEUp;
                    i += 1u;
                }
            }
            _ => panic!("classify: unhandled type")
        }
    }

    fn fixup(ty: Type, cls: &mut [RegClass]) {
        let mut i = 0u;
        let ty_kind = ty.kind();
        let e = cls.len();
        if cls.len() > 2u && (ty_kind == Struct || ty_kind == Array || ty_kind == Vector) {
            if cls[i].is_sse() {
                i += 1u;
                while i < e {
                    if cls[i] != SSEUp {
                        all_mem(cls);
                        return;
                    }
                    i += 1u;
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
                    while i != e && cls[i] == SSEUp { i += 1u; }
                } else if cls[i] == X87 {
                    i += 1;
                    while i != e && cls[i] == X87Up { i += 1u; }
                } else {
                    i += 1;
                }
            }
        }
    }

    let words = (ty_size(ty) + 7) / 8;
    let mut cls: Vec<_> = repeat(NoClass).take(words).collect();
    if words > 4 {
        all_mem(cls.as_mut_slice());
        return cls;
    }
    classify(ty, cls.as_mut_slice(), 0, 0);
    fixup(ty, cls.as_mut_slice());
    return cls;
}

fn llreg_ty(ccx: &CrateContext, cls: &[RegClass]) -> Type {
    fn llvec_len(cls: &[RegClass]) -> uint {
        let mut len = 1u;
        for c in cls.iter() {
            if *c != SSEUp {
                break;
            }
            len += 1u;
        }
        return len;
    }

    let mut tys = Vec::new();
    let mut i = 0u;
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
                    _ => unreachable!(),
                };
                let vec_len = llvec_len(&cls[i + 1u..]);
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
            _ => panic!("llregtype: unhandled class")
        }
        i += 1u;
    }
    if tys.len() == 1 && tys[0].kind() == Vector {
        // if the type contains only a vector, pass it as that vector.
        tys[0]
    } else {
        Type::struct_(ccx, tys.as_slice(), false)
    }
}

pub fn compute_abi_info(ccx: &CrateContext,
                        atys: &[Type],
                        rty: Type,
                        ret_def: bool) -> FnType {
    fn x86_64_ty<F>(ccx: &CrateContext,
                    ty: Type,
                    is_mem_cls: F,
                    ind_attr: Attribute)
                    -> ArgType where
        F: FnOnce(&[RegClass]) -> bool,
    {
        if !ty.is_reg_ty() {
            let cls = classify_ty(ty);
            if is_mem_cls(cls.as_slice()) {
                ArgType::indirect(ty, Some(ind_attr))
            } else {
                ArgType::direct(ty,
                                Some(llreg_ty(ccx, cls.as_slice())),
                                None,
                                None)
            }
        } else {
            let attr = if ty == Type::i1(ccx) { Some(ZExtAttribute) } else { None };
            ArgType::direct(ty, None, None, attr)
        }
    }

    let mut arg_tys = Vec::new();
    for t in atys.iter() {
        let ty = x86_64_ty(ccx, *t, |cls| cls.is_pass_byval(), ByValAttribute);
        arg_tys.push(ty);
    }

    let ret_ty = if ret_def {
        x86_64_ty(ccx, rty, |cls| cls.is_ret_bysret(), StructRetAttribute)
    } else {
        ArgType::direct(Type::void(ccx), None, None, None)
    };

    return FnType {
        arg_tys: arg_tys,
        ret_ty: ret_ty,
    };
}
