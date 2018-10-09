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

pub use llvm::Type;

use llvm;
use llvm::{Bool, False, True};
use context::CodegenCx;
use value::Value;
use rustc_codegen_ssa::interfaces::*;

use rustc::util::nodemap::FxHashMap;
use rustc::ty::{Ty, TyCtxt};
use rustc::ty::layout::TyLayout;
use rustc_target::abi::call::{CastTarget, FnType, Reg};
use rustc_data_structures::small_c_str::SmallCStr;
use common;
use rustc_codegen_ssa::common::TypeKind;
use type_of::LayoutLlvmExt;
use abi::{LlvmType, FnTypeExt};

use std::fmt;
use std::cell::RefCell;

use libc::c_uint;

impl PartialEq for Type {
    fn eq(&self, other: &Self) -> bool {
        self as *const _ == other as *const _
    }
}

impl fmt::Debug for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&llvm::build_string(|s| unsafe {
            llvm::LLVMRustWriteTypeToString(self, s);
        }).expect("non-UTF8 type description from LLVM"))
    }
}

impl<'ll> CodegenObject for &'ll Type {}

impl BaseTypeMethods<'ll, 'tcx> for CodegenCx<'ll, 'tcx, &'ll Value> {

    fn type_void(&self) -> &'ll Type {
        unsafe {
            llvm::LLVMVoidTypeInContext(&self.llcx)
        }
    }

    fn type_metadata(&self) -> &'ll Type {
        unsafe {
            llvm::LLVMRustMetadataTypeInContext(self.llcx)
        }
    }

    fn type_i1(&self) -> &'ll Type {
        unsafe {
            llvm::LLVMInt1TypeInContext(&self.llcx)
        }
    }

    fn type_i8(&self) -> &'ll Type {
        unsafe {
            llvm::LLVMInt8TypeInContext(&self.llcx)
        }
    }


    fn type_i16(&self) -> &'ll Type {
        unsafe {

            llvm::LLVMInt16TypeInContext(&self.llcx)
        }
    }

    fn type_i32(&self) -> &'ll Type {
        unsafe {
            llvm::LLVMInt32TypeInContext(&self.llcx)
        }
    }

    fn type_i64(&self) -> &'ll Type {
        unsafe {
            llvm::LLVMInt64TypeInContext(&self.llcx)
        }
    }

    fn type_i128(&self) -> &'ll Type {
        unsafe {
            llvm::LLVMIntTypeInContext(&self.llcx, 128)
        }
    }

    fn type_ix(&self, num_bits: u64) -> &'ll Type {
        unsafe {
            llvm::LLVMIntTypeInContext(&self.llcx, num_bits as c_uint)
        }
    }

    fn type_isize(&self) -> &'ll Type {
        self.isize_ty
    }

    fn type_f32(&self) -> &'ll Type {
        unsafe {
            llvm::LLVMFloatTypeInContext(&self.llcx)
        }
    }

    fn type_f64(&self) -> &'ll Type {
        unsafe {
            llvm::LLVMDoubleTypeInContext(&self.llcx)
        }
    }

    fn type_x86_mmx(&self) -> &'ll Type {
        unsafe {
            llvm::LLVMX86MMXTypeInContext(&self.llcx)
        }
    }

    fn type_func(
        &self,
        args: &[&'ll Type],
        ret: &'ll Type
    ) -> &'ll Type {
        unsafe {
            llvm::LLVMFunctionType(ret, args.as_ptr(),
                                   args.len() as c_uint, False)
        }
    }

    fn type_variadic_func(
        &self,
        args: &[&'ll Type],
        ret: &'ll Type
    ) -> &'ll Type {
        unsafe {
            llvm::LLVMFunctionType(ret, args.as_ptr(),
                                   args.len() as c_uint, True)
        }
    }

    fn type_struct(
        &self,
        els: &[&'ll Type],
        packed: bool
    ) -> &'ll Type {
        unsafe {
            llvm::LLVMStructTypeInContext(&self.llcx, els.as_ptr(),
                                          els.len() as c_uint,
                                          packed as Bool)
        }
    }

    fn type_named_struct(&self, name: &str) -> &'ll Type {
        let name = SmallCStr::new(name);
        unsafe {
            llvm::LLVMStructCreateNamed(&self.llcx, name.as_ptr())
        }
    }


    fn type_array(&self, ty: &'ll Type, len: u64) -> &'ll Type {
        unsafe {
            llvm::LLVMRustArrayType(ty, len)
        }
    }

    fn type_vector(&self, ty: &'ll Type, len: u64) -> &'ll Type {
        unsafe {
            llvm::LLVMVectorType(ty, len as c_uint)
        }
    }

    fn type_kind(&self, ty: &'ll Type) -> TypeKind {
        unsafe {
            llvm::LLVMRustGetTypeKind(ty).to_generic()
        }
    }

    fn set_struct_body(&self, ty: &'ll Type, els: &[&'ll Type], packed: bool) {
        unsafe {
            llvm::LLVMStructSetBody(ty, els.as_ptr(),
                                    els.len() as c_uint, packed as Bool)
        }
    }

    fn type_ptr_to(&self, ty: &'ll Type) -> &'ll Type {
        ty.ptr_to()
    }

    fn element_type(&self, ty: &'ll Type) -> &'ll Type {
        unsafe {
            llvm::LLVMGetElementType(ty)
        }
    }

    fn vector_length(&self, ty: &'ll Type) -> usize {
        unsafe {
            llvm::LLVMGetVectorSize(ty) as usize
        }
    }

    fn func_params_types(&self, ty: &'ll Type) -> Vec<&'ll Type> {
        unsafe {
            let n_args = llvm::LLVMCountParamTypes(ty) as usize;
            let mut args = Vec::with_capacity(n_args);
            llvm::LLVMGetParamTypes(ty, args.as_mut_ptr());
            args.set_len(n_args);
            args
        }
    }

    fn float_width(&self, ty : &'ll Type) -> usize {
        match &self.type_kind(ty) {
            TypeKind::Float => 32,
            TypeKind::Double => 64,
            TypeKind::X86_FP80 => 80,
            TypeKind::FP128 | TypeKind::PPC_FP128 => 128,
            _ => bug!("llvm_float_width called on a non-float type")
        }
    }

    fn int_width(&self, ty: &'ll Type) -> u64 {
        unsafe {
            llvm::LLVMGetIntTypeWidth(ty) as u64
        }
    }

    fn val_ty(&self, v: &'ll Value) -> &'ll Type {
        common::val_ty(v)
    }

    fn scalar_lltypes(&self) -> &RefCell<FxHashMap<Ty<'tcx>, Self::Type>> {
        &self.scalar_lltypes
    }

    fn tcx(&self) -> &TyCtxt<'ll, 'tcx, 'tcx> {
        &self.tcx
    }
}

impl Type {
    pub fn i8_llcx(llcx: &llvm::Context) -> &Type {
        unsafe {
            llvm::LLVMInt8TypeInContext(llcx)
        }
    }

    // Creates an integer type with the given number of bits, e.g. i24
    pub fn ix_llcx(
        llcx: &llvm::Context,
        num_bits: u64
    ) -> &Type {
        unsafe {
            llvm::LLVMIntTypeInContext(llcx, num_bits as c_uint)
        }
    }

    pub fn i8p_llcx(llcx: &'ll llvm::Context) -> &'ll Type {
        Type::i8_llcx(llcx).ptr_to()
    }

    pub fn ptr_to(&self) -> &Type {
        unsafe {
            llvm::LLVMPointerType(&self, 0)
        }
    }
}

impl<'a, 'll: 'a, 'tcx: 'll> DerivedTypeMethods<'a, 'll, 'tcx> for CodegenCx<'ll, 'tcx, &'ll Value>
    {}

impl LayoutTypeMethods<'ll, 'tcx> for CodegenCx<'ll, 'tcx, &'ll Value> {
    fn backend_type(&self, ty: &TyLayout<'tcx>) -> &'ll Type {
        ty.llvm_type(&self)
    }
    fn immediate_backend_type(&self, ty: &TyLayout<'tcx>) -> &'ll Type {
        ty.immediate_llvm_type(&self)
    }
    fn is_backend_immediate(&self, ty: &TyLayout<'tcx>) -> bool {
        ty.is_llvm_immediate()
    }
    fn is_backend_scalar_pair(&self, ty: &TyLayout<'tcx>) -> bool {
        ty.is_llvm_scalar_pair()
    }
    fn backend_field_index(&self, ty: &TyLayout<'tcx>, index: usize) -> u64 {
        ty.llvm_field_index(index)
    }
    fn scalar_pair_element_backend_type<'a>(
        &self,
        ty: &TyLayout<'tcx>,
        index: usize,
        immediate: bool
    ) -> &'ll Type {
        ty.scalar_pair_element_llvm_type(&self, index, immediate)
    }
    fn cast_backend_type(&self, ty: &CastTarget) -> &'ll Type {
        ty.llvm_type(&self)
    }
    fn fn_backend_type(&self, ty: &FnType<'tcx, Ty<'tcx>>) -> &'ll Type {
        ty.llvm_type(&self)
    }
    fn reg_backend_type(&self, ty: &Reg) -> &'ll Type {
        ty.llvm_type(&self)
    }
}

impl<'a, 'll: 'a, 'tcx: 'll> TypeMethods<'a, 'll, 'tcx> for CodegenCx<'ll, 'tcx, &'ll Value> {}
