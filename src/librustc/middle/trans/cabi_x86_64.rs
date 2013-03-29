// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
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
use middle::trans::common::*;
use middle::trans::cabi::*;

use core::libc::c_uint;
use core::option;
use core::option::Option;
use core::uint;
use core::vec;

#[deriving(Eq)]
enum x86_64_reg_class {
    no_class,
    integer_class,
    sse_fs_class,
    sse_fv_class,
    sse_ds_class,
    sse_dv_class,
    sse_int_class,
    sseup_class,
    x87_class,
    x87up_class,
    complex_x87_class,
    memory_class
}

fn is_sse(++c: x86_64_reg_class) -> bool {
    return match c {
        sse_fs_class | sse_fv_class |
        sse_ds_class | sse_dv_class => true,
        _ => false
    };
}

fn is_ymm(cls: &[x86_64_reg_class]) -> bool {
    let len = vec::len(cls);
    return (len > 2u &&
         is_sse(cls[0]) &&
         cls[1] == sseup_class &&
         cls[2] == sseup_class) ||
        (len > 3u &&
         is_sse(cls[1]) &&
         cls[2] == sseup_class &&
         cls[3] == sseup_class);
}

fn classify_ty(ty: TypeRef) -> ~[x86_64_reg_class] {
    fn align(off: uint, ty: TypeRef) -> uint {
        let a = ty_align(ty);
        return (off + a - 1u) / a * a;
    }

    fn ty_align(ty: TypeRef) -> uint {
        unsafe {
            return match llvm::LLVMGetTypeKind(ty) {
                Integer => {
                    ((llvm::LLVMGetIntTypeWidth(ty) as uint) + 7) / 8
                }
                Pointer => 8,
                Float => 4,
                Double => 8,
                Struct => {
                  do vec::foldl(1, struct_tys(ty)) |a, t| {
                      uint::max(a, ty_align(*t))
                  }
                }
                Array => {
                    let elt = llvm::LLVMGetElementType(ty);
                    ty_align(elt)
                }
                _ => fail!(~"ty_size: unhandled type")
            };
        }
    }

    fn ty_size(ty: TypeRef) -> uint {
        unsafe {
            return match llvm::LLVMGetTypeKind(ty) {
                Integer => {
                    ((llvm::LLVMGetIntTypeWidth(ty) as uint) + 7) / 8
                }
                Pointer => 8,
                Float => 4,
                Double => 8,
                Struct => {
                  let size = do vec::foldl(0, struct_tys(ty)) |s, t| {
                      align(s, *t) + ty_size(*t)
                  };
                  align(size, ty)
                }
                Array => {
                  let len = llvm::LLVMGetArrayLength(ty) as uint;
                  let elt = llvm::LLVMGetElementType(ty);
                  let eltsz = ty_size(elt);
                  len * eltsz
                }
                _ => fail!(~"ty_size: unhandled type")
            };
        }
    }

    fn all_mem(cls: &mut [x86_64_reg_class]) {
        for uint::range(0, cls.len()) |i| {
            cls[i] = memory_class;
        }
    }

    fn unify(cls: &mut [x86_64_reg_class],
             i: uint,
             newv: x86_64_reg_class) {
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

    fn classify_struct(tys: &[TypeRef],
                       cls: &mut [x86_64_reg_class], i: uint,
                       off: uint) {
        let mut field_off = off;
        for vec::each(tys) |ty| {
            field_off = align(field_off, *ty);
            classify(*ty, cls, i, field_off);
            field_off += ty_size(*ty);
        }
    }

    fn classify(ty: TypeRef,
                cls: &mut [x86_64_reg_class], ix: uint,
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

            match llvm::LLVMGetTypeKind(ty) as int {
                8 /* integer */ |
                12 /* pointer */ => {
                    unify(cls, ix + off / 8u, integer_class);
                }
                2 /* float */ => {
                    if off % 8u == 4u {
                        unify(cls, ix + off / 8u, sse_fv_class);
                    } else {
                        unify(cls, ix + off / 8u, sse_fs_class);
                    }
                }
                3 /* double */ => {
                    unify(cls, ix + off / 8u, sse_ds_class);
                }
                10 /* struct */ => {
                    classify_struct(struct_tys(ty), cls, ix, off);
                }
                11 /* array */ => {
                    let elt = llvm::LLVMGetElementType(ty);
                    let eltsz = ty_size(elt);
                    let len = llvm::LLVMGetArrayLength(ty) as uint;
                    let mut i = 0u;
                    while i < len {
                        classify(elt, cls, ix, off + i * eltsz);
                        i += 1u;
                    }
                }
                _ => fail!(~"classify: unhandled type")
            }
        }
    }

    fn fixup(ty: TypeRef, cls: &mut [x86_64_reg_class]) {
        unsafe {
            let mut i = 0u;
            let llty = llvm::LLVMGetTypeKind(ty) as int;
            let e = vec::len(cls);
            if vec::len(cls) > 2u &&
               (llty == 10 /* struct */ ||
                llty == 11 /* array */) {
                if is_sse(cls[i]) {
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
                    } else if is_sse(cls[i]) {
                        i += 1;
                        while cls[i] == sseup_class { i += 1u; }
                    } else if cls[i] == x87_class {
                        i += 1;
                        while cls[i] == x87up_class { i += 1u; }
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

fn llreg_ty(cls: &[x86_64_reg_class]) -> TypeRef {
    fn llvec_len(cls: &[x86_64_reg_class]) -> uint {
        let mut len = 1u;
        for vec::each(cls) |c| {
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
        let e = vec::len(cls);
        while i < e {
            match cls[i] {
                integer_class => {
                    tys.push(T_i64());
                }
                sse_fv_class => {
                    let vec_len = llvec_len(vec::tailn(cls, i + 1u)) * 2u;
                    let vec_ty = llvm::LLVMVectorType(T_f32(),
                                                      vec_len as c_uint);
                    tys.push(vec_ty);
                    i += vec_len;
                    loop;
                }
                sse_fs_class => {
                    tys.push(T_f32());
                }
                sse_ds_class => {
                    tys.push(T_f64());
                }
                _ => fail!(~"llregtype: unhandled class")
            }
            i += 1u;
        }
        return T_struct(tys);
    }
}

fn x86_64_tys(atys: &[TypeRef],
              rty: TypeRef,
              ret_def: bool) -> FnType {
    fn is_reg_ty(ty: TypeRef) -> bool {
        unsafe {
            return match llvm::LLVMGetTypeKind(ty) as int {
                8 /* integer */ |
                12 /* pointer */ |
                2 /* float */ |
                3 /* double */ => true,
                _ => false
            };
        }
    }

    fn is_pass_byval(cls: &[x86_64_reg_class]) -> bool {
        return cls.len() > 0 &&
            (cls[0] == memory_class ||
             cls[0] == x87_class ||
             cls[0] == complex_x87_class);
    }

    fn is_ret_bysret(cls: &[x86_64_reg_class]) -> bool {
        return cls.len() > 0 && cls[0] == memory_class;
    }

    fn x86_64_ty(ty: TypeRef,
                 is_mem_cls: &fn(cls: &[x86_64_reg_class]) -> bool,
                 attr: Attribute) -> (LLVMType, Option<Attribute>) {
        let mut cast = false;
        let mut ty_attr = option::None;
        let mut llty = ty;
        if !is_reg_ty(ty) {
            let cls = classify_ty(ty);
            if is_mem_cls(cls) {
                llty = T_ptr(ty);
                ty_attr = option::Some(attr);
            } else {
                cast = true;
                llty = llreg_ty(cls);
            }
        }
        return (LLVMType { cast: cast, ty: llty }, ty_attr);
    }

    let mut arg_tys = ~[];
    let mut attrs = ~[];
    for vec::each(atys) |t| {
        let (ty, attr) = x86_64_ty(*t, is_pass_byval, ByValAttribute);
        arg_tys.push(ty);
        attrs.push(attr);
    }
    let mut (ret_ty, ret_attr) = x86_64_ty(rty, is_ret_bysret,
                                       StructRetAttribute);
    let sret = ret_attr.is_some();
    if sret {
        arg_tys = vec::append(~[ret_ty], arg_tys);
        ret_ty = LLVMType {
                   cast:  false,
                   ty: T_void()
                 };
        attrs = vec::append(~[ret_attr], attrs);
    } else if !ret_def {
        ret_ty = LLVMType {
                   cast: false,
                   ty: T_void()
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
                    atys: &[TypeRef],
                    rty: TypeRef,
                    ret_def: bool) -> FnType {
        return x86_64_tys(atys, rty, ret_def);
    }
}

pub fn x86_64_abi_info() -> @ABIInfo {
    return @X86_64_ABIInfo as @ABIInfo;
}
