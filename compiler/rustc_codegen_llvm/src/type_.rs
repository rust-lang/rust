use std::borrow::Borrow;
use std::hash::{Hash, Hasher};
use std::{fmt, ptr};

use libc::c_uint;
use rustc_abi::{AddressSpace, Align, Float, Integer, Reg, Size};
use rustc_codegen_ssa::traits::*;
use rustc_data_structures::small_c_str::SmallCStr;
use rustc_middle::ty::layout::TyAndLayout;
use rustc_middle::ty::{self, Ty};
use rustc_span::bug;
use rustc_target::callconv::{CastTarget, FnAbi};

use crate::abi::{FnAbiLlvmExt, LlvmType};
use crate::common;
use crate::context::{CodegenCx, GenericCx, SCx};
use crate::llvm::{self, FALSE, TRUE, ToLlvmBool, Type, TypeKind, Value};
use crate::type_of::LayoutLlvmExt;

impl PartialEq for Type {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self, other)
    }
}

impl Eq for Type {}

impl Hash for Type {
    fn hash<H: Hasher>(&self, state: &mut H) {
        ptr::hash(self, state);
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

impl<'ll> CodegenCx<'ll, '_> {}
impl<'ll, CX: Borrow<SCx<'ll>>> GenericCx<'ll, CX> {
    pub(crate) fn type_named_struct(&self, name: &str) -> &'ll Type {
        let name = SmallCStr::new(name);
        unsafe { llvm::LLVMStructCreateNamed(self.llcx(), name.as_ptr()) }
    }

    pub(crate) fn set_struct_body(&self, ty: &'ll Type, els: &[&'ll Type], packed: bool) {
        unsafe {
            llvm::LLVMStructSetBody(ty, els.as_ptr(), els.len() as c_uint, packed.to_llvm_bool())
        }
    }
    pub(crate) fn type_void(&self) -> &'ll Type {
        unsafe { llvm::LLVMVoidTypeInContext(self.llcx()) }
    }

    ///x Creates an integer type with the given number of bits, e.g., i24
    pub(crate) fn type_ix(&self, num_bits: u64) -> &'ll Type {
        llvm::LLVMIntTypeInContext(self.llcx(), num_bits as c_uint)
    }

    pub(crate) fn type_vector(&self, ty: &'ll Type, count: u64) -> &'ll Type {
        unsafe { llvm::LLVMVectorType(ty, count as c_uint) }
    }

    pub(crate) fn type_scalable_vector(&self, ty: &'ll Type, count: u64) -> &'ll Type {
        unsafe { llvm::LLVMScalableVectorType(ty, count as c_uint) }
    }

    pub(crate) fn add_func(&self, name: &str, ty: &'ll Type) -> &'ll Value {
        let name = SmallCStr::new(name);
        unsafe { llvm::LLVMAddFunction(self.llmod(), name.as_ptr(), ty) }
    }

    pub(crate) fn get_return_type(&self, ty: &'ll Type) -> &'ll Type {
        unsafe { llvm::LLVMGetReturnType(ty) }
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

    pub(crate) fn func_is_variadic(&self, ty: &'ll Type) -> bool {
        unsafe { llvm::LLVMIsFunctionVarArg(ty).is_true() }
    }

    pub(crate) fn struct_element_types(&self, ty: &'ll Type) -> Vec<&'ll Type> {
        unsafe {
            let n_args = llvm::LLVMCountStructElementTypes(ty) as usize;
            let mut args = Vec::with_capacity(n_args);
            llvm::LLVMGetStructElementTypes(ty, args.as_mut_ptr());
            args.set_len(n_args);
            args
        }
    }
}
impl<'ll, 'tcx> CodegenCx<'ll, 'tcx> {
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
}

impl<'ll, CX: Borrow<SCx<'ll>>> GenericCx<'ll, CX> {
    pub(crate) fn llcx(&self) -> llvm::Context<'ll> {
        (**self).borrow().llcx
    }

    pub(crate) fn llmod(&self) -> llvm::Module<'ll> {
        (**self).borrow().llmod
    }

    pub(crate) fn isize_ty(&self) -> &'ll Type {
        (**self).borrow().isize_ty
    }

    pub(crate) fn type_variadic_func(&self, args: &[&'ll Type], ret: &'ll Type) -> &'ll Type {
        unsafe { llvm::LLVMFunctionType(ret, args.as_ptr(), args.len() as c_uint, TRUE) }
    }

    pub(crate) fn type_i1(&self) -> &'ll Type {
        unsafe { llvm::LLVMInt1TypeInContext(self.llcx()) }
    }

    pub(crate) fn type_struct(&self, els: &[&'ll Type]) -> &'ll Type {
        unsafe {
            llvm::LLVMStructTypeInContext(self.llcx(), els.as_ptr(), els.len() as c_uint, FALSE)
        }
    }

    pub(crate) fn type_packed_struct(&self, els: &[&'ll Type]) -> &'ll Type {
        unsafe {
            llvm::LLVMStructTypeInContext(self.llcx(), els.as_ptr(), els.len() as c_uint, TRUE)
        }
    }

    pub(crate) fn type_bf16(&self) -> &'ll Type {
        unsafe { llvm::LLVMBFloatTypeInContext(self.llcx()) }
    }

    pub(crate) fn type_array(&self, ty: &'ll Type, len: u64) -> &'ll Type {
        unsafe { llvm::LLVMArrayType2(ty, len) }
    }

    pub(crate) fn type_x86_fp80(&self) -> &'ll Type {
        unsafe { llvm::LLVMX86FP80TypeInContext(self.llcx()) }
    }

    /// Prefer more specific methods (e.g. `int_width`, `float_type`, `vector_element_type`) when
    /// possible.
    pub(crate) fn type_kind(&self, ty: &'ll Type) -> TypeKind {
        llvm::LLVMGetTypeKind(ty).to_rust()
    }
}

impl<'ll, CX: Borrow<SCx<'ll>>> BaseTypeCodegenMethods for GenericCx<'ll, CX> {
    fn type_i8(&self) -> &'ll Type {
        unsafe { llvm::LLVMInt8TypeInContext(self.llcx()) }
    }

    fn type_i16(&self) -> &'ll Type {
        unsafe { llvm::LLVMInt16TypeInContext(self.llcx()) }
    }

    fn type_i32(&self) -> &'ll Type {
        unsafe { llvm::LLVMInt32TypeInContext(self.llcx()) }
    }

    fn type_i64(&self) -> &'ll Type {
        unsafe { llvm::LLVMInt64TypeInContext(self.llcx()) }
    }

    fn type_i128(&self) -> &'ll Type {
        self.type_ix(128)
    }

    fn type_isize(&self) -> &'ll Type {
        self.isize_ty()
    }

    fn type_f16(&self) -> &'ll Type {
        unsafe { llvm::LLVMHalfTypeInContext(self.llcx()) }
    }

    fn type_f16b(&self) -> &'ll Type {
        unsafe { llvm::LLVMBFloatTypeInContext(self.llcx()) }
    }

    fn type_f32(&self) -> &'ll Type {
        unsafe { llvm::LLVMFloatTypeInContext(self.llcx()) }
    }

    fn type_f64(&self) -> &'ll Type {
        unsafe { llvm::LLVMDoubleTypeInContext(self.llcx()) }
    }

    fn type_f128(&self) -> &'ll Type {
        unsafe { llvm::LLVMFP128TypeInContext(self.llcx()) }
    }

    fn type_ppcf128(&self) -> &'ll Type {
        unsafe { llvm::LLVMPPCFP128TypeInContext(self.llcx()) }
    }

    fn type_func(&self, args: &[&'ll Type], ret: &'ll Type) -> &'ll Type {
        unsafe { llvm::LLVMFunctionType(ret, args.as_ptr(), args.len() as c_uint, FALSE) }
    }

    fn type_ptr(&self) -> &'ll Type {
        llvm_type_ptr(self.llcx())
    }

    fn type_ptr_ext(&self, address_space: AddressSpace) -> &'ll Type {
        llvm_type_ptr_in_address_space(self.llcx(), address_space)
    }

    fn vector_element_type_and_length(&self, ty: Self::Type) -> Option<(Self::Type, u64)> {
        match self.type_kind(ty) {
            TypeKind::Vector => {
                Some(unsafe { (llvm::LLVMGetElementType(ty), llvm::LLVMGetVectorSize(ty).into()) })
            }
            _ => None,
        }
    }

    fn float_type(&self, ty: &'ll Type) -> Option<Float> {
        match self.type_kind(ty) {
            TypeKind::BFloat => Some(Float::F16B),
            TypeKind::Half => Some(Float::F16),
            TypeKind::Float => Some(Float::F32),
            TypeKind::Double => Some(Float::F64),
            TypeKind::FP128 => Some(Float::F128),
            TypeKind::X86_FP80 => bug!("x87_f80 is not supported yet"),
            TypeKind::PPC_FP128 => Some(Float::PpcF128),
            _ => None,
        }
    }

    fn int_width(&self, ty: &'ll Type) -> Option<u64> {
        match self.type_kind(ty) {
            TypeKind::Integer => Some(unsafe { llvm::LLVMGetIntTypeWidth(ty).into() }),
            _ => None,
        }
    }

    fn val_ty(&self, v: &'ll Value) -> &'ll Type {
        common::val_ty(v)
    }
}

pub(crate) fn llvm_type_ptr<'ll>(llcx: llvm::Context<'ll>) -> &'ll Type {
    llvm_type_ptr_in_address_space(llcx, AddressSpace::ZERO)
}

pub(crate) fn llvm_type_ptr_in_address_space<'ll>(
    llcx: llvm::Context<'ll>,
    addr_space: AddressSpace,
) -> &'ll Type {
    llvm::LLVMPointerTypeInContext(llcx, addr_space.0)
}

impl<'ll, 'tcx> LayoutTypeCodegenMethods<'tcx> for CodegenCx<'ll, 'tcx> {
    fn backend_type(&self, layout: TyAndLayout<'tcx>) -> &'ll Type {
        layout.llvm_type(self)
    }
    fn immediate_backend_type(&self, layout: TyAndLayout<'tcx>) -> &'ll Type {
        layout.immediate_llvm_type(self)
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
    fn add_type_metadata(&self, function: &'ll Value, typeid: &[u8]) {
        let typeid_metadata = self.create_metadata(typeid);
        let v = [llvm::LLVMValueAsMetadata(self.const_usize(0)), typeid_metadata];
        self.global_add_metadata_node(function, llvm::MD_type, &v);
    }

    fn set_type_metadata(&self, function: &'ll Value, typeid: &[u8]) {
        let typeid_metadata = self.create_metadata(typeid);
        let v = [llvm::LLVMValueAsMetadata(self.const_usize(0)), typeid_metadata];
        self.global_set_metadata_node(function, llvm::MD_type, &v);
    }

    fn add_kcfi_type_metadata(&self, function: &'ll Value, kcfi_typeid: u32) {
        let kcfi_type_metadata = [llvm::LLVMValueAsMetadata(self.const_u32(kcfi_typeid))];
        self.global_add_metadata_node(function, llvm::MD_kcfi_type, &kcfi_type_metadata);
    }

    fn set_kcfi_type_metadata(&self, function: &'ll Value, kcfi_typeid: u32) {
        let kcfi_type_metadata = [llvm::LLVMValueAsMetadata(self.const_u32(kcfi_typeid))];
        self.global_set_metadata_node(function, llvm::MD_kcfi_type, &kcfi_type_metadata);
    }
}
