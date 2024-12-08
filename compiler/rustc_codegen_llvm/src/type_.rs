use std::{fmt, ptr};

use libc::{c_char, c_uint};
use rustc_abi::{AddressSpace, Align, Integer, Size};
use rustc_codegen_ssa::common::TypeKind;
use rustc_codegen_ssa::traits::*;
use rustc_data_structures::small_c_str::SmallCStr;
use rustc_middle::bug;
use rustc_middle::ty::layout::TyAndLayout;
use rustc_middle::ty::{self, Ty};
use rustc_target::callconv::{CastTarget, FnAbi, Reg};

use crate::abi::{FnAbiLlvmExt, LlvmType};
use crate::context::CodegenCx;
pub(crate) use crate::llvm::Type;
use crate::llvm::{Bool, False, Metadata, True};
use crate::type_of::LayoutLlvmExt;
use crate::value::Value;
use crate::{common, llvm};

impl PartialEq for Type {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self, other)
    }
}

impl fmt::Debug for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(
            &llvm::build_string(|s| unsafe {
                llvm::LLVMRustWriteTypeToString(self, s);
            })
            .expect("non-UTF8 type description from LLVM"),
        )
    }
}

impl<'ll> CodegenCx<'ll, '_> {
    pub(crate) fn type_named_struct(&self, name: &str) -> &'ll Type {
        let name = SmallCStr::new(name);
        unsafe { llvm::LLVMStructCreateNamed(self.llcx, name.as_ptr()) }
    }

    pub(crate) fn set_struct_body(&self, ty: &'ll Type, els: &[&'ll Type], packed: bool) {
        unsafe { llvm::LLVMStructSetBody(ty, els.as_ptr(), els.len() as c_uint, packed as Bool) }
    }

    pub(crate) fn type_void(&self) -> &'ll Type {
        unsafe { llvm::LLVMVoidTypeInContext(self.llcx) }
    }

    pub(crate) fn type_token(&self) -> &'ll Type {
        unsafe { llvm::LLVMTokenTypeInContext(self.llcx) }
    }

    pub(crate) fn type_metadata(&self) -> &'ll Type {
        unsafe { llvm::LLVMMetadataTypeInContext(self.llcx) }
    }

    ///x Creates an integer type with the given number of bits, e.g., i24
    pub(crate) fn type_ix(&self, num_bits: u64) -> &'ll Type {
        unsafe { llvm::LLVMIntTypeInContext(self.llcx, num_bits as c_uint) }
    }

    pub(crate) fn type_vector(&self, ty: &'ll Type, len: u64) -> &'ll Type {
        unsafe { llvm::LLVMVectorType(ty, len as c_uint) }
    }

    pub(crate) fn func_params_types(&self, ty: &'ll Type) -> Vec<&'ll Type> {
        unsafe {
            let n_args = llvm::LLVMCountParamTypes(ty) as usize;
            let mut args = Vec::with_capacity(n_args);
            llvm::LLVMGetParamTypes(ty, args.as_mut_ptr());
            args.set_len(n_args);
            args
        }
    }

    pub(crate) fn type_bool(&self) -> &'ll Type {
        self.type_i8()
    }

    pub(crate) fn type_int_from_ty(&self, t: ty::IntTy) -> &'ll Type {
        match t {
            ty::IntTy::Isize => self.type_isize(),
            ty::IntTy::I8 => self.type_i8(),
            ty::IntTy::I16 => self.type_i16(),
            ty::IntTy::I32 => self.type_i32(),
            ty::IntTy::I64 => self.type_i64(),
            ty::IntTy::I128 => self.type_i128(),
        }
    }

    pub(crate) fn type_uint_from_ty(&self, t: ty::UintTy) -> &'ll Type {
        match t {
            ty::UintTy::Usize => self.type_isize(),
            ty::UintTy::U8 => self.type_i8(),
            ty::UintTy::U16 => self.type_i16(),
            ty::UintTy::U32 => self.type_i32(),
            ty::UintTy::U64 => self.type_i64(),
            ty::UintTy::U128 => self.type_i128(),
        }
    }

    pub(crate) fn type_float_from_ty(&self, t: ty::FloatTy) -> &'ll Type {
        match t {
            ty::FloatTy::F16 => self.type_f16(),
            ty::FloatTy::F32 => self.type_f32(),
            ty::FloatTy::F64 => self.type_f64(),
            ty::FloatTy::F128 => self.type_f128(),
        }
    }

    /// Return an LLVM type that has at most the required alignment,
    /// and exactly the required size, as a best-effort padding array.
    pub(crate) fn type_padding_filler(&self, size: Size, align: Align) -> &'ll Type {
        let unit = Integer::approximate_align(self, align);
        let size = size.bytes();
        let unit_size = unit.size().bytes();
        assert_eq!(size % unit_size, 0);
        self.type_array(self.type_from_integer(unit), size / unit_size)
    }

    pub(crate) fn type_variadic_func(&self, args: &[&'ll Type], ret: &'ll Type) -> &'ll Type {
        unsafe { llvm::LLVMFunctionType(ret, args.as_ptr(), args.len() as c_uint, True) }
    }

    pub(crate) fn type_i1(&self) -> &'ll Type {
        unsafe { llvm::LLVMInt1TypeInContext(self.llcx) }
    }

    pub(crate) fn type_struct(&self, els: &[&'ll Type], packed: bool) -> &'ll Type {
        unsafe {
            llvm::LLVMStructTypeInContext(
                self.llcx,
                els.as_ptr(),
                els.len() as c_uint,
                packed as Bool,
            )
        }
    }
}

impl<'ll, 'tcx> BaseTypeCodegenMethods<'tcx> for CodegenCx<'ll, 'tcx> {
    fn type_i8(&self) -> &'ll Type {
        unsafe { llvm::LLVMInt8TypeInContext(self.llcx) }
    }

    fn type_i16(&self) -> &'ll Type {
        unsafe { llvm::LLVMInt16TypeInContext(self.llcx) }
    }

    fn type_i32(&self) -> &'ll Type {
        unsafe { llvm::LLVMInt32TypeInContext(self.llcx) }
    }

    fn type_i64(&self) -> &'ll Type {
        unsafe { llvm::LLVMInt64TypeInContext(self.llcx) }
    }

    fn type_i128(&self) -> &'ll Type {
        unsafe { llvm::LLVMIntTypeInContext(self.llcx, 128) }
    }

    fn type_isize(&self) -> &'ll Type {
        self.isize_ty
    }

    fn type_f16(&self) -> &'ll Type {
        unsafe { llvm::LLVMHalfTypeInContext(self.llcx) }
    }

    fn type_f32(&self) -> &'ll Type {
        unsafe { llvm::LLVMFloatTypeInContext(self.llcx) }
    }

    fn type_f64(&self) -> &'ll Type {
        unsafe { llvm::LLVMDoubleTypeInContext(self.llcx) }
    }

    fn type_f128(&self) -> &'ll Type {
        unsafe { llvm::LLVMFP128TypeInContext(self.llcx) }
    }

    fn type_func(&self, args: &[&'ll Type], ret: &'ll Type) -> &'ll Type {
        unsafe { llvm::LLVMFunctionType(ret, args.as_ptr(), args.len() as c_uint, False) }
    }

    fn type_kind(&self, ty: &'ll Type) -> TypeKind {
        unsafe { llvm::LLVMRustGetTypeKind(ty).to_generic() }
    }

    fn type_ptr(&self) -> &'ll Type {
        self.type_ptr_ext(AddressSpace::DATA)
    }

    fn type_ptr_ext(&self, address_space: AddressSpace) -> &'ll Type {
        unsafe { llvm::LLVMPointerTypeInContext(self.llcx, address_space.0) }
    }

    fn element_type(&self, ty: &'ll Type) -> &'ll Type {
        match self.type_kind(ty) {
            TypeKind::Array | TypeKind::Vector => unsafe { llvm::LLVMGetElementType(ty) },
            TypeKind::Pointer => bug!("element_type is not supported for opaque pointers"),
            other => bug!("element_type called on unsupported type {other:?}"),
        }
    }

    fn vector_length(&self, ty: &'ll Type) -> usize {
        unsafe { llvm::LLVMGetVectorSize(ty) as usize }
    }

    fn float_width(&self, ty: &'ll Type) -> usize {
        match self.type_kind(ty) {
            TypeKind::Half => 16,
            TypeKind::Float => 32,
            TypeKind::Double => 64,
            TypeKind::X86_FP80 => 80,
            TypeKind::FP128 | TypeKind::PPC_FP128 => 128,
            other => bug!("llvm_float_width called on a non-float type {other:?}"),
        }
    }

    fn int_width(&self, ty: &'ll Type) -> u64 {
        unsafe { llvm::LLVMGetIntTypeWidth(ty) as u64 }
    }

    fn val_ty(&self, v: &'ll Value) -> &'ll Type {
        common::val_ty(v)
    }

    fn type_array(&self, ty: &'ll Type, len: u64) -> &'ll Type {
        unsafe { llvm::LLVMArrayType2(ty, len) }
    }
}

impl Type {
    /// Creates an integer type with the given number of bits, e.g., i24
    pub fn ix_llcx(llcx: &llvm::Context, num_bits: u64) -> &Type {
        unsafe { llvm::LLVMIntTypeInContext(llcx, num_bits as c_uint) }
    }

    pub fn ptr_llcx(llcx: &llvm::Context) -> &Type {
        unsafe { llvm::LLVMPointerTypeInContext(llcx, AddressSpace::DATA.0) }
    }
}

impl<'ll, 'tcx> LayoutTypeCodegenMethods<'tcx> for CodegenCx<'ll, 'tcx> {
    fn backend_type(&self, layout: TyAndLayout<'tcx>) -> &'ll Type {
        layout.llvm_type(self)
    }
    fn immediate_backend_type(&self, layout: TyAndLayout<'tcx>) -> &'ll Type {
        layout.immediate_llvm_type(self)
    }
    fn is_backend_immediate(&self, layout: TyAndLayout<'tcx>) -> bool {
        layout.is_llvm_immediate()
    }
    fn is_backend_scalar_pair(&self, layout: TyAndLayout<'tcx>) -> bool {
        layout.is_llvm_scalar_pair()
    }
    fn scalar_pair_element_backend_type(
        &self,
        layout: TyAndLayout<'tcx>,
        index: usize,
        immediate: bool,
    ) -> &'ll Type {
        layout.scalar_pair_element_llvm_type(self, index, immediate)
    }
    fn cast_backend_type(&self, ty: &CastTarget) -> &'ll Type {
        ty.llvm_type(self)
    }
    fn fn_decl_backend_type(&self, fn_abi: &FnAbi<'tcx, Ty<'tcx>>) -> &'ll Type {
        fn_abi.llvm_type(self)
    }
    fn fn_ptr_backend_type(&self, fn_abi: &FnAbi<'tcx, Ty<'tcx>>) -> &'ll Type {
        fn_abi.ptr_to_llvm_type(self)
    }
    fn reg_backend_type(&self, ty: &Reg) -> &'ll Type {
        ty.llvm_type(self)
    }
}

impl<'ll, 'tcx> TypeMembershipCodegenMethods<'tcx> for CodegenCx<'ll, 'tcx> {
    fn add_type_metadata(&self, function: &'ll Value, typeid: String) {
        let typeid_metadata = self.typeid_metadata(typeid).unwrap();
        unsafe {
            let v = [llvm::LLVMValueAsMetadata(self.const_usize(0)), typeid_metadata];
            llvm::LLVMRustGlobalAddMetadata(
                function,
                llvm::MD_type as c_uint,
                llvm::LLVMMDNodeInContext2(self.llcx, v.as_ptr(), v.len()),
            )
        }
    }

    fn set_type_metadata(&self, function: &'ll Value, typeid: String) {
        let typeid_metadata = self.typeid_metadata(typeid).unwrap();
        unsafe {
            let v = [llvm::LLVMValueAsMetadata(self.const_usize(0)), typeid_metadata];
            llvm::LLVMGlobalSetMetadata(
                function,
                llvm::MD_type as c_uint,
                llvm::LLVMMDNodeInContext2(self.llcx, v.as_ptr(), v.len()),
            )
        }
    }

    fn typeid_metadata(&self, typeid: String) -> Option<&'ll Metadata> {
        Some(unsafe {
            llvm::LLVMMDStringInContext2(self.llcx, typeid.as_ptr() as *const c_char, typeid.len())
        })
    }

    fn add_kcfi_type_metadata(&self, function: &'ll Value, kcfi_typeid: u32) {
        let kcfi_type_metadata = self.const_u32(kcfi_typeid);
        unsafe {
            llvm::LLVMRustGlobalAddMetadata(
                function,
                llvm::MD_kcfi_type as c_uint,
                llvm::LLVMMDNodeInContext2(
                    self.llcx,
                    &llvm::LLVMValueAsMetadata(kcfi_type_metadata),
                    1,
                ),
            )
        }
    }

    fn set_kcfi_type_metadata(&self, function: &'ll Value, kcfi_typeid: u32) {
        let kcfi_type_metadata = self.const_u32(kcfi_typeid);
        unsafe {
            llvm::LLVMGlobalSetMetadata(
                function,
                llvm::MD_kcfi_type as c_uint,
                llvm::LLVMMDNodeInContext2(
                    self.llcx,
                    &llvm::LLVMValueAsMetadata(kcfi_type_metadata),
                    1,
                ),
            )
        }
    }
}
