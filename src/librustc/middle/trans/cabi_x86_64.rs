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

use lib::llvm::{llvm, TypeRef, Integer, Pointer, Float, Double};
use lib::llvm::{Struct, Array, Attribute};
use lib::llvm::{StructRetAttribute, ByValAttribute};
use lib::llvm::struct_tys;
use lib::llvm::True;
use middle::trans::common::*;
use middle::trans::cabi::*;

use core::libc::c_uint;
use core::option;
use core::option::Option;
use core::uint;
use core::vec;

#[deriving(Eq)]
enum RegClass {
    NoClass,
    Integer,
    SSEFs,
    SSEFv,
    SSEDs,
    SSEDv,
    SSEInt,
    SSEUp,
    X87,
    X87Up,
    ComplexX87,
    Memory
}

impl Type {
    fn is_reg_ty(&self) -> bool {
        match ty.kind() {
            Integer | Pointer | Float | Double => true,
            _ => false
        }
    }
}

impl RegClass {
    fn is_sse(&self) -> bool {
        match *self {
            SSEFs | SSEFv | SSEDs | SSEDv => true,
            _ => false
        }
    }
}

impl<'self> ClassList for &'self [RegClass] {
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

fn classify_ty(ty: Type) -> ~[RegClass] {
    fn align(off: uint, ty: Type) -> uint {
        let a = ty_align(ty);
        return (off + a - 1u) / a * a;
    }

    fn ty_align(ty: Type) -> uint {
        unsafe {
            match ty.kind() {
                Integer => {
                    ((llvm::LLVMGetIntTypeWidth(ty.to_ref()) as uint) + 7) / 8
                }
                Pointer => 8,
                Float => 4,
                Double => 8,
                Struct => {
                  if ty.is_packed() {
                    1
                  } else {
                    let str_tys = ty.field_types();
                    str_tys.iter().fold(1, |a, t| uint::max(a, ty_align(*t)))
                  }
                }
                Array => {
                    let elt = ty.element_type();
                    ty_align(elt)
                }
                _ => fail!("ty_size: unhandled type")
            };
        }
    }

    fn ty_size(ty: TypeRef) -> uint {
        unsafe {
            match ty.kind() {
                Integer => {
                    ((llvm::LLVMGetIntTypeWidth(ty) as uint) + 7) / 8
                }
                Pointer => 8,
                Float => 4,
                Double => 8,
                Struct => {
                    if ty.is_packed() {
                        let str_tys = ty.field_types();
                        str_tys.iter().fold(0, |s, t| s + ty_size(*t))
                    } else {
                        let str_tys = ty.field_types();
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
                _ => fail!("ty_size: unhandled type")
            }
        }
    }

    fn all_mem(cls: &mut [RegClass]) {
        for uint::range(0, cls.len()) |i| {
            cls[i] = memory_class;
        }
    }

    fn unify(cls: &mut [RegClass],
             i: uint,
             newv: RegClass) {
        if cls[i] == newv {
            return;
        } else if cls[i] == no_class {
            cls[i] = newv;
        } else if newv == no_class {
            return;
        } else if cls[i] == memory_class || newv == memory_class {
            cls[i] = memory_class;
        } else if cls[i] == integer_class || newv == integer_class {
            cls[i] = integer_class;
        } else if cls[i] == x87_class ||
                  cls[i] == x87up_class ||
                  cls[i] == complex_x87_class ||
                  newv == x87_class ||
                  newv == x87up_class ||
                  newv == complex_x87_class {
            cls[i] = memory_class;
        } else {
            cls[i] = newv;
        }
    }

    fn classify_struct(tys: &[Type],
                       cls: &mut [RegClass], i: uint,
                       off: uint) {
        let mut field_off = off;
        for tys.each |ty| {
            field_off = align(field_off, *ty);
            classify(*ty, cls, i, field_off);
            field_off += ty_size(*ty);
        }
    }

    fn classify(ty: Type,
                cls: &mut [RegClass], ix: uint,
                off: uint) {
        unsafe {
            let t_align = ty_align(ty);
            let t_size = ty_size(ty);

            let misalign = off % t_align;
            if misalign != 0u {
                let mut i = off / 8u;
                let e = (off + t_size + 7u) / 8u;
                while i < e {
                    unify(cls, ix + i, memory_class);
                    i += 1u;
                }
                return;
            }

            match ty.kind() {
                Integer |
                Pointer => {
                    unify(cls, ix + off / 8u, integer_class);
                }
                Float => {
                    if off % 8u == 4u {
                        unify(cls, ix + off / 8u, sse_fv_class);
                    } else {
                        unify(cls, ix + off / 8u, sse_fs_class);
                    }
                }
                Double => {
                    unify(cls, ix + off / 8u, sse_ds_class);
                }
                Struct => {
                    classify_struct(ty.field_types(), cls, ix, off);
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
                _ => fail!("classify: unhandled type")
            }
        }
    }

    fn fixup(ty: Type, cls: &mut [RegClass]) {
        unsafe {
            let mut i = 0u;
            let ty_kind = ty.kind();
            let e = cls.len();
            if cls.len() > 2u &&
               (ty_kind == Struct ||
                ty_kind == Array) {
                if cls[i].is_sse() {
                    i += 1u;
                    while i < e {
                        if cls[i] != sseup_class {
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
                    if cls[i] == memory_class {
                        all_mem(cls);
                        return;
                    }
                    if cls[i] == x87up_class {
                        // for darwin
                        // cls[i] = sse_ds_class;
                        all_mem(cls);
                        return;
                    }
                    if cls[i] == sseup_class {
                        cls[i] = sse_int_class;
                    } else if cls[i].is_sse() {
                        i += 1;
                        while i != e && cls[i] == sseup_class { i += 1u; }
                    } else if cls[i] == x87_class {
                        i += 1;
                        while i != e && cls[i] == x87up_class { i += 1u; }
                    } else {
                        i += 1;
                    }
                }
            }
        }
    }

    let words = (ty_size(ty) + 7) / 8;
    let mut cls = vec::from_elem(words, no_class);
    if words > 4 {
        all_mem(cls);
        let cls = cls;
        return cls;
    }
    classify(ty, cls, 0, 0);
    fixup(ty, cls);
    return cls;
}

fn llreg_ty(cls: &[RegClass]) -> Type {
    fn llvec_len(cls: &[RegClass]) -> uint {
        let mut len = 1u;
        for cls.each |c| {
            if *c != sseup_class {
                break;
            }
            len += 1u;
        }
        return len;
    }

    unsafe {
        let mut tys = ~[];
        let mut i = 0u;
        let e = cls.len();
        while i < e {
            match cls[i] {
                integer_class => {
                    tys.push(Type::i64());
                }
                sse_fv_class => {
                    let vec_len = llvec_len(vec::tailn(cls, i + 1u)) * 2u;
                    let vec_ty = Type::vector(Type::f32(), vec_len);
                    tys.push(vec_ty);
                    i += vec_len;
                    loop;
                }
                sse_fs_class => {
                    tys.push(Type::f32());
                }
                sse_ds_class => {
                    tys.push(Type::f64());
                }
                _ => fail!("llregtype: unhandled class")
            }
            i += 1u;
        }
        return Type::struct_(tys, false);
    }
}

fn x86_64_tys(atys: &[Type],
              rty: Type,
              ret_def: bool) -> FnType {

    fn x86_64_ty(ty: Type,
                 is_mem_cls: &fn(cls: &[RegClass]) -> bool,
                 attr: Attribute) -> (LLVMType, Option<Attribute>) {
        let (cast, attr, ty) = if !ty.is_reg_ty() {
            let cls = classify_ty(ty);
            if is_mem_cls(cls) {
                (false, option::Some(attr), ty.ptr_to())
            } else {
                (true, option::None, llreg_ty(cls))
            }
        };
        return (LLVMType { cast: cast, ty: ty }, attr);
    }

    let mut arg_tys = ~[];
    let mut attrs = ~[];
    for atys.each |t| {
        let (ty, attr) = x86_64_ty(*t, |cls| cls.is_pass_byval(), ByValAttribute);
        arg_tys.push(ty);
        attrs.push(attr);
    }
    let mut (ret_ty, ret_attr) = x86_64_ty(rty, |cls| cls.is_ret_bysret(),
                                       StructRetAttribute);
    let sret = ret_attr.is_some();
    if sret {
        arg_tys = vec::append(~[ret_ty], arg_tys);
        ret_ty = LLVMType {
                   cast:  false,
                   ty: Type::void()
                 };
        attrs = vec::append(~[ret_attr], attrs);
    } else if !ret_def {
        ret_ty = LLVMType {
                   cast: false,
                   ty: Type::void()
                 };
    }
    return FnType {
        arg_tys: arg_tys,
        ret_ty: ret_ty,
        attrs: attrs,
        sret: sret
    };
}

enum X86_64_ABIInfo { X86_64_ABIInfo }

impl ABIInfo for X86_64_ABIInfo {
    fn compute_info(&self,
                    atys: &[Type],
                    rty: Type,
                    ret_def: bool) -> FnType {
        return x86_64_tys(atys, rty, ret_def);
    }
}

pub fn abi_info() -> @ABIInfo {
    return @X86_64_ABIInfo as @ABIInfo;
}
