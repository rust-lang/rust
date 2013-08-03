// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lib::llvm::{llvm, Integer, Pointer, Float, Double, Struct, Array};
use lib::llvm::{Attribute, StructRetAttribute};
use middle::trans::cabi::{ABIInfo, FnType, LLVMType};

use middle::trans::type_::Type;

use std::num;
use std::option::{Option, None, Some};

fn align_up_to(off: uint, a: uint) -> uint {
    return (off + a - 1u) / a * a;
}

fn align(off: uint, ty: Type) -> uint {
    let a = ty_align(ty);
    return align_up_to(off, a);
}

fn ty_align(ty: Type) -> uint {
    match ty.kind() {
        Integer => {
            unsafe {
                ((llvm::LLVMGetIntTypeWidth(ty.to_ref()) as uint) + 7) / 8
            }
        }
        Pointer => 4,
        Float => 4,
        Double => 8,
        Struct => {
            if ty.is_packed() {
                1
            } else {
                let str_tys = ty.field_types();
                str_tys.iter().fold(1, |a, t| num::max(a, ty_align(*t)))
            }
        }
        Array => {
            let elt = ty.element_type();
            ty_align(elt)
        }
        _ => fail!("ty_align: unhandled type")
    }
}

fn ty_size(ty: Type) -> uint {
    match ty.kind() {
        Integer => {
            unsafe {
                ((llvm::LLVMGetIntTypeWidth(ty.to_ref()) as uint) + 7) / 8
            }
        }
        Pointer => 4,
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

fn classify_ret_ty(ty: Type) -> (LLVMType, Option<Attribute>) {
    if is_reg_ty(ty) {
        return (LLVMType { cast: false, ty: ty }, None);
    }
    let size = ty_size(ty);
    if size <= 4 {
        let llty = if size <= 1 {
            Type::i8()
        } else if size <= 2 {
            Type::i16()
        } else {
            Type::i32()
        };
        return (LLVMType { cast: true, ty: llty }, None);
    }
    (LLVMType { cast: false, ty: ty.ptr_to() }, Some(StructRetAttribute))
}

fn classify_arg_ty(ty: Type) -> (LLVMType, Option<Attribute>) {
    if is_reg_ty(ty) {
        return (LLVMType { cast: false, ty: ty }, None);
    }
    let align = ty_align(ty);
    let size = ty_size(ty);
    let llty = if align <= 4 {
        Type::array(&Type::i32(), ((size + 3) / 4) as u64)
    } else {
        Type::array(&Type::i64(), ((size + 7) / 8) as u64)
    };
    (LLVMType { cast: true, ty: llty }, None)
}

fn is_reg_ty(ty: Type) -> bool {
    match ty.kind() {
        Integer
        | Pointer
        | Float
        | Double => true,
        _ => false
    }
}

enum ARM_ABIInfo { ARM_ABIInfo }

impl ABIInfo for ARM_ABIInfo {
    fn compute_info(&self,
                    atys: &[Type],
                    rty: Type,
                    ret_def: bool) -> FnType {
        let mut arg_tys = ~[];
        let mut attrs = ~[];
        foreach &aty in atys.iter() {
            let (ty, attr) = classify_arg_ty(aty);
            arg_tys.push(ty);
            attrs.push(attr);
        }

        let (ret_ty, ret_attr) = if ret_def {
            classify_ret_ty(rty)
        } else {
            (LLVMType { cast: false, ty: Type::void() }, None)
        };

        let mut ret_ty = ret_ty;

        let sret = ret_attr.is_some();
        if sret {
            arg_tys.unshift(ret_ty);
            attrs.unshift(ret_attr);
            ret_ty = LLVMType { cast: false, ty: Type::void() };
        }

        return FnType {
            arg_tys: arg_tys,
            ret_ty: ret_ty,
            attrs: attrs,
            sret: sret
        };
    }
}

pub fn abi_info() -> @ABIInfo {
    return @ARM_ABIInfo as @ABIInfo;
}
