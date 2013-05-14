// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::libc::c_uint;
use lib::llvm::{llvm, TypeRef, Integer, Pointer, Float, Double};
use lib::llvm::{Struct, Array, Attribute};
use lib::llvm::{StructRetAttribute};
use lib::llvm::True;
use middle::trans::common::*;
use middle::trans::cabi::*;

fn align_up_to(off: uint, a: uint) -> uint {
    return (off + a - 1u) / a * a;
}

fn align(off: uint, ty: TypeRef) -> uint {
    let a = ty_align(ty);
    return align_up_to(off, a);
}

fn struct_tys(ty: TypeRef) -> ~[TypeRef] {
    unsafe {
        let n = llvm::LLVMCountStructElementTypes(ty);
    if (n == 0) {
        return ~[];
    }
        let mut elts = vec::from_elem(n as uint, ptr::null());
        llvm::LLVMGetStructElementTypes(ty,
            ptr::to_mut_unsafe_ptr(&mut elts[0]));
        return elts;
    }
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
            _ => fail!("ty_size: unhandled type")
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
    return if is_reg_ty(ty) {
        (LLVMType { cast: false, ty: ty }, None)
    } else {
        (LLVMType { cast: false, ty: T_ptr(ty) }, Some(StructRetAttribute))
    };
}

fn classify_arg_ty(ty: TypeRef,
                   offset: &mut uint) -> (LLVMType, Option<Attribute>) {
    let orig_offset = *offset;
    let size = ty_size(ty) * 8;
    let mut align = ty_align(ty);

    align = uint::min(uint::max(align, 4), 8);
    *offset = align_up_to(*offset, align);
    *offset += align_up_to(size, align * 8) / 8;

    let padding = padding_ty(align, orig_offset);
    return if !is_reg_ty(ty) {
        (LLVMType {
            cast: true,
            ty: struct_ty(ty, padding, true)
        }, None)
    } else if padding.is_some() {
        (LLVMType {
            cast: true,
            ty: struct_ty(ty, padding, false)
        }, None)
    } else {
        (LLVMType { cast: false, ty: ty }, None)
    };
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

fn padding_ty(align: uint, offset: uint) -> Option<TypeRef> {
    if ((align - 1 ) & offset) > 0 {
        return Some(T_i32());
    }

    return None;
}

fn coerce_to_int(size: uint) -> ~[TypeRef] {
    let int_ty = T_i32();
    let mut args = ~[];

    let mut n = size / 32;
    while n > 0 {
        args.push(int_ty);
        n -= 1;
    }

    let r = size % 32;
    if r > 0 {
        unsafe {
            args.push(llvm::LLVMIntType(r as c_uint))
        }
    }

    return args;
}

fn struct_ty(ty: TypeRef,
             padding: Option<TypeRef>,
             coerce: bool) -> TypeRef {
    let size = ty_size(ty) * 8;
    let mut fields = padding.map_default(~[], |p| ~[*p]);

    if coerce {
        fields = vec::append(fields, coerce_to_int(size));
    } else {
        fields.push(ty);
    }

    return T_struct(fields, false);
}

enum MIPS_ABIInfo { MIPS_ABIInfo }

impl ABIInfo for MIPS_ABIInfo {
    fn compute_info(&self,
                    atys: &[TypeRef],
                    rty: TypeRef,
                    ret_def: bool) -> FnType {
        let mut (ret_ty, ret_attr) = if ret_def {
            classify_ret_ty(rty)
        } else {
            (LLVMType { cast: false, ty: T_void() }, None)
        };

        let sret = ret_attr.is_some();
        let mut arg_tys = ~[];
        let mut attrs = ~[];
        let mut offset = if sret { 4 } else { 0 };

        for atys.each() |aty| {
            let (ty, attr) = classify_arg_ty(*aty, &mut offset);
            arg_tys.push(ty);
            attrs.push(attr);
        };

        if sret {
            arg_tys = vec::append(~[ret_ty], arg_tys);
            attrs = vec::append(~[ret_attr], attrs);
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
    return @MIPS_ABIInfo as @ABIInfo;
}
