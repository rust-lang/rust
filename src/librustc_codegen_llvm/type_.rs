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
use llvm::{Bool, False, True, TypeKind};

use context::CodegenCx;
use value::Value;
use interfaces::TypeMethods;

use syntax::ast;
use rustc::ty::layout::{self, Align, Size};
use rustc_data_structures::small_c_str::SmallCStr;
use back::write;

use std::fmt;

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

impl TypeMethods for CodegenCx<'ll, 'tcx, &'ll Value> {

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

    // Creates an integer type with the given number of bits, e.g. i24
    fn type_ix(&self, num_bits: u64) -> &'ll Type {
        unsafe {
            llvm::LLVMIntTypeInContext(&self.llcx, num_bits as c_uint)
        }
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
            llvm::LLVMRustGetTypeKind(ty)
        }
    }

    fn set_struct_body(&self, ty: &'ll Type, els: &[&'ll Type], packed: bool) {
        unsafe {
            llvm::LLVMStructSetBody(ty, els.as_ptr(),
                                    els.len() as c_uint, packed as Bool)
        }
    }

    fn type_ptr_to(&self, ty: &'ll Type) -> &'ll Type {
        unsafe {
            llvm::LLVMPointerType(ty, 0)
        }
    }

    fn element_type(&self, ty: &'ll Type) -> &'ll Type {
        unsafe {
            llvm::LLVMGetElementType(ty)
        }
    }

    /// Return the number of elements in `self` if it is a LLVM vector type.
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
            TypeKind::FP128 | TypeKind::PPc_FP128 => 128,
            _ => bug!("llvm_float_width called on a non-float type")
        }
    }

    /// Retrieve the bit width of the integer type `self`.
    fn int_width(&self, ty: &'ll Type) -> u64 {
        unsafe {
            llvm::LLVMGetIntTypeWidth(ty) as u64
        }
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

    pub fn i8p_llcx(cx : &write::CodegenContext<'ll>, llcx: &'ll llvm::Context) -> &'ll Type {
        cx.type_ptr_to(Type::i8_llcx(llcx))
    }
}

impl CodegenCx<'ll, 'tcx, &'ll Value> {

    pub fn type_bool(&self) -> &'ll Type {
        &self.type_i8()
    }

    pub fn type_char(&self) -> &'ll Type {
        &self.type_i32()
    }

    pub fn type_i8p(&self) -> &'ll Type {
        &self.type_ptr_to(&self.type_i8())
    }

    pub fn type_isize(&self) -> &'ll Type {
        &self.isize_ty
    }

    pub fn type_int(&self) -> &'ll Type {
        match &self.sess().target.target.target_c_int_width[..] {
            "16" => &self.type_i16(),
            "32" => &self.type_i32(),
            "64" => &self.type_i64(),
            width => bug!("Unsupported target_c_int_width: {}", width),
        }
    }

    pub fn type_int_from_ty(
        &self,
        t: ast::IntTy
    ) -> &'ll Type {
        match t {
            ast::IntTy::Isize => &self.isize_ty,
            ast::IntTy::I8 => &self.type_i8(),
            ast::IntTy::I16 => &self.type_i16(),
            ast::IntTy::I32 => &self.type_i32(),
            ast::IntTy::I64 => &self.type_i64(),
            ast::IntTy::I128 => &self.type_i128(),
        }
    }

    pub fn type_uint_from_ty(
        &self,
        t: ast::UintTy
    ) -> &'ll Type {
        match t {
            ast::UintTy::Usize => &self.isize_ty,
            ast::UintTy::U8 => &self.type_i8(),
            ast::UintTy::U16 => &self.type_i16(),
            ast::UintTy::U32 => &self.type_i32(),
            ast::UintTy::U64 => &self.type_i64(),
            ast::UintTy::U128 => &self.type_i128(),
        }
    }

    pub fn type_float_from_ty(
        &self,
        t: ast::FloatTy
    ) -> &'ll Type {
        match t {
            ast::FloatTy::F32 => &self.type_f32(),
            ast::FloatTy::F64 => &self.type_f64(),
        }
    }

    pub fn type_from_integer(&self, i: layout::Integer) -> &'ll Type {
        use rustc::ty::layout::Integer::*;
        match i {
            I8 => &self.type_i8(),
            I16 => &self.type_i16(),
            I32 => &self.type_i32(),
            I64 => &self.type_i64(),
            I128 => &self.type_i128(),
        }
    }

    /// Return a LLVM type that has at most the required alignment,
    /// as a conservative approximation for unknown pointee types.
    pub fn type_pointee_for_abi_align(&self, align: Align) -> &'ll Type {
        // FIXME(eddyb) We could find a better approximation if ity.align < align.
        let ity = layout::Integer::approximate_abi_align(self, align);
        &self.type_from_integer(ity)
    }

    /// Return a LLVM type that has at most the required alignment,
    /// and exactly the required size, as a best-effort padding array.
    pub fn type_padding_filler(
        &self,
        size: Size,
        align: Align
    ) -> &'ll Type {
        let unit = layout::Integer::approximate_abi_align(self, align);
        let size = size.bytes();
        let unit_size = unit.size().bytes();
        assert_eq!(size % unit_size, 0);
        &self.type_array(&self.type_from_integer(unit), size / unit_size)
    }
}
