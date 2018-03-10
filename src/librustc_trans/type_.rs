// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_upper_case_globals)]

use llvm;
use llvm::{ContextRef, TypeRef, Bool, False, True, TypeKind};
use llvm::{Float, Double, X86_FP80, PPC_FP128, FP128};

use context::CodegenCx;

use syntax::ast;
use rustc::ty::layout::{self, Align, Size};

use std::ffi::CString;
use std::fmt;
use std::mem;
use std::ptr;

use libc::c_uint;

#[derive(Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Type {
    rf: TypeRef
}

impl fmt::Debug for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&llvm::build_string(|s| unsafe {
            llvm::LLVMRustWriteTypeToString(self.to_ref(), s);
        }).expect("non-UTF8 type description from LLVM"))
    }
}

macro_rules! ty {
    ($e:expr) => ( Type::from_ref(unsafe { $e }))
}

/// Wrapper for LLVM TypeRef
impl Type {
    #[inline(always)]
    pub fn from_ref(r: TypeRef) -> Type {
        Type {
            rf: r
        }
    }

    #[inline(always)] // So it doesn't kill --opt-level=0 builds of the compiler
    pub fn to_ref(&self) -> TypeRef {
        self.rf
    }

    pub fn to_ref_slice(slice: &[Type]) -> &[TypeRef] {
        unsafe { mem::transmute(slice) }
    }

    pub fn void(cx: &CodegenCx) -> Type {
        ty!(llvm::LLVMVoidTypeInContext(cx.llcx))
    }

    pub fn metadata(cx: &CodegenCx) -> Type {
        ty!(llvm::LLVMRustMetadataTypeInContext(cx.llcx))
    }

    pub fn i1(cx: &CodegenCx) -> Type {
        ty!(llvm::LLVMInt1TypeInContext(cx.llcx))
    }

    pub fn i8(cx: &CodegenCx) -> Type {
        ty!(llvm::LLVMInt8TypeInContext(cx.llcx))
    }

    pub fn i8_llcx(llcx: ContextRef) -> Type {
        ty!(llvm::LLVMInt8TypeInContext(llcx))
    }

    pub fn i16(cx: &CodegenCx) -> Type {
        ty!(llvm::LLVMInt16TypeInContext(cx.llcx))
    }

    pub fn i32(cx: &CodegenCx) -> Type {
        ty!(llvm::LLVMInt32TypeInContext(cx.llcx))
    }

    pub fn i64(cx: &CodegenCx) -> Type {
        ty!(llvm::LLVMInt64TypeInContext(cx.llcx))
    }

    pub fn i128(cx: &CodegenCx) -> Type {
        ty!(llvm::LLVMIntTypeInContext(cx.llcx, 128))
    }

    // Creates an integer type with the given number of bits, e.g. i24
    pub fn ix(cx: &CodegenCx, num_bits: u64) -> Type {
        ty!(llvm::LLVMIntTypeInContext(cx.llcx, num_bits as c_uint))
    }

    pub fn f32(cx: &CodegenCx) -> Type {
        ty!(llvm::LLVMFloatTypeInContext(cx.llcx))
    }

    pub fn f64(cx: &CodegenCx) -> Type {
        ty!(llvm::LLVMDoubleTypeInContext(cx.llcx))
    }

    pub fn bool(cx: &CodegenCx) -> Type {
        Type::i8(cx)
    }

    pub fn char(cx: &CodegenCx) -> Type {
        Type::i32(cx)
    }

    pub fn i8p(cx: &CodegenCx) -> Type {
        Type::i8(cx).ptr_to()
    }

    pub fn i8p_llcx(llcx: ContextRef) -> Type {
        Type::i8_llcx(llcx).ptr_to()
    }

    pub fn isize(cx: &CodegenCx) -> Type {
        match &cx.tcx.sess.target.target.target_pointer_width[..] {
            "16" => Type::i16(cx),
            "32" => Type::i32(cx),
            "64" => Type::i64(cx),
            tws => bug!("Unsupported target word size for int: {}", tws),
        }
    }

    pub fn c_int(cx: &CodegenCx) -> Type {
        match &cx.tcx.sess.target.target.target_c_int_width[..] {
            "16" => Type::i16(cx),
            "32" => Type::i32(cx),
            "64" => Type::i64(cx),
            width => bug!("Unsupported target_c_int_width: {}", width),
        }
    }

    pub fn int_from_ty(cx: &CodegenCx, t: ast::IntTy) -> Type {
        match t {
            ast::IntTy::Isize => cx.isize_ty,
            ast::IntTy::I8 => Type::i8(cx),
            ast::IntTy::I16 => Type::i16(cx),
            ast::IntTy::I32 => Type::i32(cx),
            ast::IntTy::I64 => Type::i64(cx),
            ast::IntTy::I128 => Type::i128(cx),
        }
    }

    pub fn uint_from_ty(cx: &CodegenCx, t: ast::UintTy) -> Type {
        match t {
            ast::UintTy::Usize => cx.isize_ty,
            ast::UintTy::U8 => Type::i8(cx),
            ast::UintTy::U16 => Type::i16(cx),
            ast::UintTy::U32 => Type::i32(cx),
            ast::UintTy::U64 => Type::i64(cx),
            ast::UintTy::U128 => Type::i128(cx),
        }
    }

    pub fn float_from_ty(cx: &CodegenCx, t: ast::FloatTy) -> Type {
        match t {
            ast::FloatTy::F32 => Type::f32(cx),
            ast::FloatTy::F64 => Type::f64(cx),
        }
    }

    pub fn func(args: &[Type], ret: &Type) -> Type {
        let slice: &[TypeRef] = Type::to_ref_slice(args);
        ty!(llvm::LLVMFunctionType(ret.to_ref(), slice.as_ptr(),
                                   args.len() as c_uint, False))
    }

    pub fn variadic_func(args: &[Type], ret: &Type) -> Type {
        let slice: &[TypeRef] = Type::to_ref_slice(args);
        ty!(llvm::LLVMFunctionType(ret.to_ref(), slice.as_ptr(),
                                   args.len() as c_uint, True))
    }

    pub fn struct_(cx: &CodegenCx, els: &[Type], packed: bool) -> Type {
        let els: &[TypeRef] = Type::to_ref_slice(els);
        ty!(llvm::LLVMStructTypeInContext(cx.llcx, els.as_ptr(),
                                          els.len() as c_uint,
                                          packed as Bool))
    }

    pub fn named_struct(cx: &CodegenCx, name: &str) -> Type {
        let name = CString::new(name).unwrap();
        ty!(llvm::LLVMStructCreateNamed(cx.llcx, name.as_ptr()))
    }


    pub fn array(ty: &Type, len: u64) -> Type {
        ty!(llvm::LLVMRustArrayType(ty.to_ref(), len))
    }

    pub fn vector(ty: &Type, len: u64) -> Type {
        ty!(llvm::LLVMVectorType(ty.to_ref(), len as c_uint))
    }

    pub fn kind(&self) -> TypeKind {
        unsafe {
            llvm::LLVMRustGetTypeKind(self.to_ref())
        }
    }

    pub fn set_struct_body(&mut self, els: &[Type], packed: bool) {
        let slice: &[TypeRef] = Type::to_ref_slice(els);
        unsafe {
            llvm::LLVMStructSetBody(self.to_ref(), slice.as_ptr(),
                                    els.len() as c_uint, packed as Bool)
        }
    }

    pub fn ptr_to(&self) -> Type {
        ty!(llvm::LLVMPointerType(self.to_ref(), 0))
    }

    pub fn element_type(&self) -> Type {
        unsafe {
            Type::from_ref(llvm::LLVMGetElementType(self.to_ref()))
        }
    }

    /// Return the number of elements in `self` if it is a LLVM vector type.
    pub fn vector_length(&self) -> usize {
        unsafe {
            llvm::LLVMGetVectorSize(self.to_ref()) as usize
        }
    }

    pub fn func_params(&self) -> Vec<Type> {
        unsafe {
            let n_args = llvm::LLVMCountParamTypes(self.to_ref()) as usize;
            let mut args = vec![Type { rf: ptr::null_mut() }; n_args];
            llvm::LLVMGetParamTypes(self.to_ref(),
                                    args.as_mut_ptr() as *mut TypeRef);
            args
        }
    }

    pub fn float_width(&self) -> usize {
        match self.kind() {
            Float => 32,
            Double => 64,
            X86_FP80 => 80,
            FP128 | PPC_FP128 => 128,
            _ => bug!("llvm_float_width called on a non-float type")
        }
    }

    /// Retrieve the bit width of the integer type `self`.
    pub fn int_width(&self) -> u64 {
        unsafe {
            llvm::LLVMGetIntTypeWidth(self.to_ref()) as u64
        }
    }

    pub fn from_integer(cx: &CodegenCx, i: layout::Integer) -> Type {
        use rustc::ty::layout::Integer::*;
        match i {
            I8 => Type::i8(cx),
            I16 => Type::i16(cx),
            I32 => Type::i32(cx),
            I64 => Type::i64(cx),
            I128 => Type::i128(cx),
        }
    }

    /// Return a LLVM type that has at most the required alignment,
    /// as a conservative approximation for unknown pointee types.
    pub fn pointee_for_abi_align(cx: &CodegenCx, align: Align) -> Type {
        // FIXME(eddyb) We could find a better approximation if ity.align < align.
        let ity = layout::Integer::approximate_abi_align(cx, align);
        Type::from_integer(cx, ity)
    }

    /// Return a LLVM type that has at most the required alignment,
    /// and exactly the required size, as a best-effort padding array.
    pub fn padding_filler(cx: &CodegenCx, size: Size, align: Align) -> Type {
        let unit = layout::Integer::approximate_abi_align(cx, align);
        let size = size.bytes();
        let unit_size = unit.size().bytes();
        assert_eq!(size % unit_size, 0);
        Type::array(&Type::from_integer(cx, unit), size / unit_size)
    }

    pub fn x86_mmx(cx: &CodegenCx) -> Type {
        ty!(llvm::LLVMX86MMXTypeInContext(cx.llcx))
    }
}
