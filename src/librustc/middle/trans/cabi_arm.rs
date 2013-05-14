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
use lib::llvm::struct_tys;
use lib::llvm::TypeRef;
use lib::llvm::{Attribute, StructRetAttribute};
use lib::llvm::True;
use middle::trans::cabi::{ABIInfo, FnType, LLVMType};
use middle::trans::common::{T_i8, T_i16, T_i32, T_i64};
use middle::trans::common::{T_array, T_ptr, T_void};

use core::option::{Option, None, Some};
use core::uint;
use core::vec;

fn align_up_to(off: uint, a: uint) -> uint {
    return (off + a - 1u) / a * a;
}

fn align(off: uint, ty: TypeRef) -> uint {
    let a = ty_align(ty);
    return align_up_to(off, a);
}

fn ty_align(ty: TypeRef) -> uint {
    unsafe {
        return match llvm::LLVMGetTypeKind(ty) {
            Integer => {
                ((llvm::LLVMGetIntTypeWidth(ty) as uint) + 7) / 8
            }
            Pointer => 4,
            Float => 4,
            Double => 8,
            Struct => {
                if llvm::LLVMIsPackedStruct(ty) == True {
                    1
                } else {
                    do vec::foldl(1, struct_tys(ty)) |a, t| {
                        uint::max(a, ty_align(*t))
                    }
                }
            }
            Array => {
                let elt = llvm::LLVMGetElementType(ty);
                ty_align(elt)
            }
            _ => fail!("ty_align: unhandled type")
        };
    }
}

fn ty_size(ty: TypeRef) -> uint {
    unsafe {
        return match llvm::LLVMGetTypeKind(ty) {
            Integer => {
                ((llvm::LLVMGetIntTypeWidth(ty) as uint) + 7) / 8
            }
            Pointer => 4,
            Float => 4,
            Double => 8,
            Struct => {
                if llvm::LLVMIsPackedStruct(ty) == True {
                    do vec::foldl(0, struct_tys(ty)) |s, t| {
                        s + ty_size(*t)
                    }
                } else {
                    let size = do vec::foldl(0, struct_tys(ty)) |s, t| {
                        align(s, *t) + ty_size(*t)
                    };
                    align(size, ty)
                }
            }
            Array => {
                let len = llvm::LLVMGetArrayLength(ty) as uint;
                let elt = llvm::LLVMGetElementType(ty);
                let eltsz = ty_size(elt);
                len * eltsz
            }
            _ => fail!("ty_size: unhandled type")
        };
    }
}

fn classify_ret_ty(ty: TypeRef) -> (LLVMType, Option<Attribute>) {
    if is_reg_ty(ty) {
        return (LLVMType { cast: false, ty: ty }, None);
    }
    let size = ty_size(ty);
    if size <= 4 {
        let llty = if size <= 1 {
            T_i8()
        } else if size <= 2 {
            T_i16()
        } else {
            T_i32()
        };
        return (LLVMType { cast: true, ty: llty }, None);
    }
    (LLVMType { cast: false, ty: T_ptr(ty) }, Some(StructRetAttribute))
}

fn classify_arg_ty(ty: TypeRef) -> (LLVMType, Option<Attribute>) {
    if is_reg_ty(ty) {
        return (LLVMType { cast: false, ty: ty }, None);
    }
    let align = ty_align(ty);
    let size = ty_size(ty);
    let llty = if align <= 4 {
        T_array(T_i32(), (size + 3) / 4)
    } else {
        T_array(T_i64(), (size + 7) / 8)
    };
    (LLVMType { cast: true, ty: llty }, None)
}

fn is_reg_ty(ty: TypeRef) -> bool {
    unsafe {
        return match llvm::LLVMGetTypeKind(ty) {
            Integer
            | Pointer
            | Float
            | Double => true,
            _ => false
        };
    }
}

enum ARM_ABIInfo { ARM_ABIInfo }

impl ABIInfo for ARM_ABIInfo {
    fn compute_info(&self,
                    atys: &[TypeRef],
                    rty: TypeRef,
                    ret_def: bool) -> FnType {
        let mut arg_tys = ~[];
        let mut attrs = ~[];
        for atys.each |&aty| {
            let (ty, attr) = classify_arg_ty(aty);
            arg_tys.push(ty);
            attrs.push(attr);
        }

        let mut (ret_ty, ret_attr) = if ret_def {
            classify_ret_ty(rty)
        } else {
            (LLVMType { cast: false, ty: T_void() }, None)
        };

        let sret = ret_attr.is_some();
        if sret {
            arg_tys.unshift(ret_ty);
            attrs.unshift(ret_attr);
            ret_ty = LLVMType { cast: false, ty: T_void() };
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
