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
use llvm::{TypeRef, Bool, False, True, TypeKind};
use llvm::{Float, Double, X86_FP80, PPC_FP128, FP128};

use context::CrateContext;

use syntax::ast;
use rustc::ty::layout;

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

    pub fn void(ccx: &CrateContext) -> Type {
        ty!(llvm::LLVMVoidTypeInContext(ccx.llcx()))
    }

    pub fn nil(ccx: &CrateContext) -> Type {
        Type::empty_struct(ccx)
    }

    pub fn metadata(ccx: &CrateContext) -> Type {
        ty!(llvm::LLVMRustMetadataTypeInContext(ccx.llcx()))
    }

    pub fn i1(ccx: &CrateContext) -> Type {
        ty!(llvm::LLVMInt1TypeInContext(ccx.llcx()))
    }

    pub fn i8(ccx: &CrateContext) -> Type {
        ty!(llvm::LLVMInt8TypeInContext(ccx.llcx()))
    }

    pub fn i16(ccx: &CrateContext) -> Type {
        ty!(llvm::LLVMInt16TypeInContext(ccx.llcx()))
    }

    pub fn i32(ccx: &CrateContext) -> Type {
        ty!(llvm::LLVMInt32TypeInContext(ccx.llcx()))
    }

    pub fn i64(ccx: &CrateContext) -> Type {
        ty!(llvm::LLVMInt64TypeInContext(ccx.llcx()))
    }

    pub fn i128(ccx: &CrateContext) -> Type {
        ty!(llvm::LLVMIntTypeInContext(ccx.llcx(), 128))
    }

    // Creates an integer type with the given number of bits, e.g. i24
    pub fn ix(ccx: &CrateContext, num_bits: u64) -> Type {
        ty!(llvm::LLVMIntTypeInContext(ccx.llcx(), num_bits as c_uint))
    }

    pub fn f32(ccx: &CrateContext) -> Type {
        ty!(llvm::LLVMFloatTypeInContext(ccx.llcx()))
    }

    pub fn f64(ccx: &CrateContext) -> Type {
        ty!(llvm::LLVMDoubleTypeInContext(ccx.llcx()))
    }

    pub fn bool(ccx: &CrateContext) -> Type {
        Type::i8(ccx)
    }

    pub fn char(ccx: &CrateContext) -> Type {
        Type::i32(ccx)
    }

    pub fn i8p(ccx: &CrateContext) -> Type {
        Type::i8(ccx).ptr_to()
    }

    pub fn int(ccx: &CrateContext) -> Type {
        match &ccx.tcx().sess.target.target.target_pointer_width[..] {
            "16" => Type::i16(ccx),
            "32" => Type::i32(ccx),
            "64" => Type::i64(ccx),
            tws => bug!("Unsupported target word size for int: {}", tws),
        }
    }

    pub fn int_from_ty(ccx: &CrateContext, t: ast::IntTy) -> Type {
        match t {
            ast::IntTy::Is => ccx.int_type(),
            ast::IntTy::I8 => Type::i8(ccx),
            ast::IntTy::I16 => Type::i16(ccx),
            ast::IntTy::I32 => Type::i32(ccx),
            ast::IntTy::I64 => Type::i64(ccx),
            ast::IntTy::I128 => Type::i128(ccx),
        }
    }

    pub fn uint_from_ty(ccx: &CrateContext, t: ast::UintTy) -> Type {
        match t {
            ast::UintTy::Us => ccx.int_type(),
            ast::UintTy::U8 => Type::i8(ccx),
            ast::UintTy::U16 => Type::i16(ccx),
            ast::UintTy::U32 => Type::i32(ccx),
            ast::UintTy::U64 => Type::i64(ccx),
            ast::UintTy::U128 => Type::i128(ccx),
        }
    }

    pub fn float_from_ty(ccx: &CrateContext, t: ast::FloatTy) -> Type {
        match t {
            ast::FloatTy::F32 => Type::f32(ccx),
            ast::FloatTy::F64 => Type::f64(ccx),
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

    pub fn struct_(ccx: &CrateContext, els: &[Type], packed: bool) -> Type {
        let els: &[TypeRef] = Type::to_ref_slice(els);
        ty!(llvm::LLVMStructTypeInContext(ccx.llcx(), els.as_ptr(),
                                          els.len() as c_uint,
                                          packed as Bool))
    }

    pub fn named_struct(ccx: &CrateContext, name: &str) -> Type {
        let name = CString::new(name).unwrap();
        ty!(llvm::LLVMStructCreateNamed(ccx.llcx(), name.as_ptr()))
    }

    pub fn empty_struct(ccx: &CrateContext) -> Type {
        Type::struct_(ccx, &[], false)
    }

    pub fn array(ty: &Type, len: u64) -> Type {
        ty!(llvm::LLVMRustArrayType(ty.to_ref(), len))
    }

    pub fn vector(ty: &Type, len: u64) -> Type {
        ty!(llvm::LLVMVectorType(ty.to_ref(), len as c_uint))
    }

    pub fn vec(ccx: &CrateContext, ty: &Type) -> Type {
        Type::struct_(ccx,
            &[Type::array(ty, 0), Type::int(ccx)],
        false)
    }

    pub fn opaque_vec(ccx: &CrateContext) -> Type {
        Type::vec(ccx, &Type::i8(ccx))
    }

    pub fn vtable_ptr(ccx: &CrateContext) -> Type {
        Type::func(&[Type::i8p(ccx)], &Type::void(ccx)).ptr_to().ptr_to()
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

    pub fn is_aggregate(&self) -> bool {
        match self.kind() {
            TypeKind::Struct | TypeKind::Array => true,
            _ =>  false
        }
    }

    pub fn is_packed(&self) -> bool {
        unsafe {
            llvm::LLVMIsPackedStruct(self.to_ref()) == True
        }
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

    pub fn array_length(&self) -> usize {
        unsafe {
            llvm::LLVMGetArrayLength(self.to_ref()) as usize
        }
    }

    pub fn field_types(&self) -> Vec<Type> {
        unsafe {
            let n_elts = llvm::LLVMCountStructElementTypes(self.to_ref()) as usize;
            if n_elts == 0 {
                return Vec::new();
            }
            let mut elts = vec![Type { rf: ptr::null_mut() }; n_elts];
            llvm::LLVMGetStructElementTypes(self.to_ref(),
                                            elts.as_mut_ptr() as *mut TypeRef);
            elts
        }
    }

    pub fn return_type(&self) -> Type {
        ty!(llvm::LLVMGetReturnType(self.to_ref()))
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

    pub fn from_integer(cx: &CrateContext, i: layout::Integer) -> Type {
        use rustc::ty::layout::Integer::*;
        match i {
            I1 => Type::i1(cx),
            I8 => Type::i8(cx),
            I16 => Type::i16(cx),
            I32 => Type::i32(cx),
            I64 => Type::i64(cx),
            I128 => Type::i128(cx),
        }
    }

    pub fn from_primitive(ccx: &CrateContext, p: layout::Primitive) -> Type {
        match p {
            layout::Int(i) => Type::from_integer(ccx, i),
            layout::F32 => Type::f32(ccx),
            layout::F64 => Type::f64(ccx),
            layout::Pointer => bug!("It is not possible to convert Pointer directly to Type.")
        }
    }
}
