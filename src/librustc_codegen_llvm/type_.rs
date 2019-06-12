#![allow(non_upper_case_globals)]

pub use crate::llvm::Type;

use crate::llvm;
use crate::llvm::{Bool, False, True};
use crate::context::CodegenCx;
use crate::value::Value;
use rustc_codegen_ssa::traits::*;

use crate::common;
use crate::type_of::LayoutLlvmExt;
use crate::abi::{LlvmType, FnTypeLlvmExt};
use syntax::ast;
use rustc::ty::Ty;
use rustc::ty::layout::{self, Align, Size, TyLayout};
use rustc_target::abi::call::{CastTarget, FnType, Reg};
use rustc_data_structures::small_c_str::SmallCStr;
use rustc_codegen_ssa::common::TypeKind;

use std::fmt;
use std::ptr;

use libc::c_uint;

impl PartialEq for Type {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self, other)
    }
}

impl fmt::Debug for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&llvm::build_string(|s| unsafe {
            llvm::LLVMRustWriteTypeToString(self, s);
        }).expect("non-UTF8 type description from LLVM"))
    }
}

impl CodegenCx<'ll, 'tcx> {
    crate fn type_named_struct(&self, name: &str) -> &'ll Type {
        let name = SmallCStr::new(name);
        unsafe {
            llvm::LLVMStructCreateNamed(self.llcx, name.as_ptr())
        }
    }

    crate fn set_struct_body(&self, ty: &'ll Type, els: &[&'ll Type], packed: bool) {
        unsafe {
            llvm::LLVMStructSetBody(ty, els.as_ptr(),
                                    els.len() as c_uint, packed as Bool)
        }
    }

    crate fn type_void(&self) -> &'ll Type {
        unsafe {
            llvm::LLVMVoidTypeInContext(self.llcx)
        }
    }

    crate fn type_metadata(&self) -> &'ll Type {
        unsafe {
            llvm::LLVMRustMetadataTypeInContext(self.llcx)
        }
    }

    ///x Creates an integer type with the given number of bits, e.g., i24
    crate fn type_ix(&self, num_bits: u64) -> &'ll Type {
        unsafe {
            llvm::LLVMIntTypeInContext(self.llcx, num_bits as c_uint)
        }
    }

    crate fn type_x86_mmx(&self) -> &'ll Type {
        unsafe {
            llvm::LLVMX86MMXTypeInContext(self.llcx)
        }
    }

    crate fn type_vector(&self, ty: &'ll Type, len: u64) -> &'ll Type {
        unsafe {
            llvm::LLVMVectorType(ty, len as c_uint)
        }
    }

    crate fn func_params_types(&self, ty: &'ll Type) -> Vec<&'ll Type> {
        unsafe {
            let n_args = llvm::LLVMCountParamTypes(ty) as usize;
            let mut args = Vec::with_capacity(n_args);
            llvm::LLVMGetParamTypes(ty, args.as_mut_ptr());
            args.set_len(n_args);
            args
        }
    }

    crate fn type_bool(&self) -> &'ll Type {
        self.type_i8()
    }

    crate fn type_int_from_ty(&self, t: ast::IntTy) -> &'ll Type {
        match t {
            ast::IntTy::Isize => self.type_isize(),
            ast::IntTy::I8 => self.type_i8(),
            ast::IntTy::I16 => self.type_i16(),
            ast::IntTy::I32 => self.type_i32(),
            ast::IntTy::I64 => self.type_i64(),
            ast::IntTy::I128 => self.type_i128(),
        }
    }

    crate fn type_uint_from_ty(&self, t: ast::UintTy) -> &'ll Type {
        match t {
            ast::UintTy::Usize => self.type_isize(),
            ast::UintTy::U8 => self.type_i8(),
            ast::UintTy::U16 => self.type_i16(),
            ast::UintTy::U32 => self.type_i32(),
            ast::UintTy::U64 => self.type_i64(),
            ast::UintTy::U128 => self.type_i128(),
        }
    }

    crate fn type_float_from_ty(&self, t: ast::FloatTy) -> &'ll Type {
        match t {
            ast::FloatTy::F32 => self.type_f32(),
            ast::FloatTy::F64 => self.type_f64(),
        }
    }

    crate fn type_pointee_for_align(&self, align: Align) -> &'ll Type {
        // FIXME(eddyb) We could find a better approximation if ity.align < align.
        let ity = layout::Integer::approximate_align(self, align);
        self.type_from_integer(ity)
    }

    /// Return a LLVM type that has at most the required alignment,
    /// and exactly the required size, as a best-effort padding array.
    crate fn type_padding_filler(&self, size: Size, align: Align) -> &'ll Type {
        let unit = layout::Integer::approximate_align(self, align);
        let size = size.bytes();
        let unit_size = unit.size().bytes();
        assert_eq!(size % unit_size, 0);
        self.type_array(self.type_from_integer(unit), size / unit_size)
    }

    crate fn type_variadic_func(
        &self,
        args: &[&'ll Type],
        ret: &'ll Type
    ) -> &'ll Type {
        unsafe {
            llvm::LLVMFunctionType(ret, args.as_ptr(),
                                   args.len() as c_uint, True)
        }
    }

    crate fn type_array(&self, ty: &'ll Type, len: u64) -> &'ll Type {
        unsafe {
            llvm::LLVMRustArrayType(ty, len)
        }
    }
}

impl BaseTypeMethods<'tcx> for CodegenCx<'ll, 'tcx> {
    fn type_i1(&self) -> &'ll Type {
        unsafe {
            llvm::LLVMInt1TypeInContext(self.llcx)
        }
    }

    fn type_i8(&self) -> &'ll Type {
        unsafe {
            llvm::LLVMInt8TypeInContext(self.llcx)
        }
    }


    fn type_i16(&self) -> &'ll Type {
        unsafe {
            llvm::LLVMInt16TypeInContext(self.llcx)
        }
    }

    fn type_i32(&self) -> &'ll Type {
        unsafe {
            llvm::LLVMInt32TypeInContext(self.llcx)
        }
    }

    fn type_i64(&self) -> &'ll Type {
        unsafe {
            llvm::LLVMInt64TypeInContext(self.llcx)
        }
    }

    fn type_i128(&self) -> &'ll Type {
        unsafe {
            llvm::LLVMIntTypeInContext(self.llcx, 128)
        }
    }

    fn type_isize(&self) -> &'ll Type {
        self.isize_ty
    }

    fn type_f32(&self) -> &'ll Type {
        unsafe {
            llvm::LLVMFloatTypeInContext(self.llcx)
        }
    }

    fn type_f64(&self) -> &'ll Type {
        unsafe {
            llvm::LLVMDoubleTypeInContext(self.llcx)
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

    fn type_struct(
        &self,
        els: &[&'ll Type],
        packed: bool
    ) -> &'ll Type {
        unsafe {
            llvm::LLVMStructTypeInContext(self.llcx, els.as_ptr(),
                                          els.len() as c_uint,
                                          packed as Bool)
        }
    }

    fn type_kind(&self, ty: &'ll Type) -> TypeKind {
        unsafe {
            llvm::LLVMRustGetTypeKind(ty).to_generic()
        }
    }

    fn type_ptr_to(&self, ty: &'ll Type) -> &'ll Type {
        assert_ne!(self.type_kind(ty), TypeKind::Function,
                   "don't call ptr_to on function types, use ptr_to_llvm_type on FnType instead");
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

    fn float_width(&self, ty: &'ll Type) -> usize {
        match self.type_kind(ty) {
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
}

impl Type {
    pub fn i8_llcx(llcx: &llvm::Context) -> &Type {
        unsafe {
            llvm::LLVMInt8TypeInContext(llcx)
        }
    }

    // Creates an integer type with the given number of bits, e.g., i24
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

    fn ptr_to(&self) -> &Type {
        unsafe {
            llvm::LLVMPointerType(&self, 0)
        }
    }
}


impl LayoutTypeMethods<'tcx> for CodegenCx<'ll, 'tcx> {
    fn backend_type(&self, layout: TyLayout<'tcx>) -> &'ll Type {
        layout.llvm_type(self)
    }
    fn immediate_backend_type(&self, layout: TyLayout<'tcx>) -> &'ll Type {
        layout.immediate_llvm_type(self)
    }
    fn is_backend_immediate(&self, layout: TyLayout<'tcx>) -> bool {
        layout.is_llvm_immediate()
    }
    fn is_backend_scalar_pair(&self, layout: TyLayout<'tcx>) -> bool {
        layout.is_llvm_scalar_pair()
    }
    fn backend_field_index(&self, layout: TyLayout<'tcx>, index: usize) -> u64 {
        layout.llvm_field_index(index)
    }
    fn scalar_pair_element_backend_type(
        &self,
        layout: TyLayout<'tcx>,
        index: usize,
        immediate: bool
    ) -> &'ll Type {
        layout.scalar_pair_element_llvm_type(self, index, immediate)
    }
    fn cast_backend_type(&self, ty: &CastTarget) -> &'ll Type {
        ty.llvm_type(self)
    }
    fn fn_ptr_backend_type(&self, ty: &FnType<'tcx, Ty<'tcx>>) -> &'ll Type {
        ty.ptr_to_llvm_type(self)
    }
    fn reg_backend_type(&self, ty: &Reg) -> &'ll Type {
        ty.llvm_type(self)
    }
}
