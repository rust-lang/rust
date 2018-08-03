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

use syntax::ast;
use rustc::ty::layout::{self, Align, Size};
use rustc_data_structures::small_c_str::SmallCStr;

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

impl Type {
    pub fn void(cx: &CodegenCx<'ll, '_, &'ll Value>) -> &'ll Type {
        unsafe {
            llvm::LLVMVoidTypeInContext(cx.llcx)
        }
    }

    pub fn metadata(cx: &CodegenCx<'ll, '_, &'ll Value>) -> &'ll Type {
        unsafe {
            llvm::LLVMRustMetadataTypeInContext(cx.llcx)
        }
    }

    pub fn i1(cx: &CodegenCx<'ll, '_, &'ll Value>) -> &'ll Type {
        unsafe {
            llvm::LLVMInt1TypeInContext(cx.llcx)
        }
    }

    pub fn i8(cx: &CodegenCx<'ll, '_, &'ll Value>) -> &'ll Type {
        unsafe {
            llvm::LLVMInt8TypeInContext(cx.llcx)
        }
    }

    pub fn i8_llcx(llcx: &llvm::Context) -> &Type {
        unsafe {
            llvm::LLVMInt8TypeInContext(llcx)
        }
    }

    pub fn i16(cx: &CodegenCx<'ll, '_, &'ll Value>) -> &'ll Type {
        unsafe {
            llvm::LLVMInt16TypeInContext(cx.llcx)
        }
    }

    pub fn i32(cx: &CodegenCx<'ll, '_, &'ll Value>) -> &'ll Type {
        unsafe {
            llvm::LLVMInt32TypeInContext(cx.llcx)
        }
    }

    pub fn i64(cx: &CodegenCx<'ll, '_, &'ll Value>) -> &'ll Type {
        unsafe {
            llvm::LLVMInt64TypeInContext(cx.llcx)
        }
    }

    pub fn i128(cx: &CodegenCx<'ll, '_, &'ll Value>) -> &'ll Type {
        unsafe {
            llvm::LLVMIntTypeInContext(cx.llcx, 128)
        }
    }

    // Creates an integer type with the given number of bits, e.g. i24
    pub fn ix(cx: &CodegenCx<'ll, '_, &'ll Value>, num_bits: u64) -> &'ll Type {
        unsafe {
            llvm::LLVMIntTypeInContext(cx.llcx, num_bits as c_uint)
        }
    }

    // Creates an integer type with the given number of bits, e.g. i24
    pub fn ix_llcx(llcx: &llvm::Context, num_bits: u64) -> &Type {
        unsafe {
            llvm::LLVMIntTypeInContext(llcx, num_bits as c_uint)
        }
    }

    pub fn f32(cx: &CodegenCx<'ll, '_, &'ll Value>) -> &'ll Type {
        unsafe {
            llvm::LLVMFloatTypeInContext(cx.llcx)
        }
    }

    pub fn f64(cx: &CodegenCx<'ll, '_, &'ll Value>) -> &'ll Type {
        unsafe {
            llvm::LLVMDoubleTypeInContext(cx.llcx)
        }
    }

    pub fn bool(cx: &CodegenCx<'ll, '_, &'ll Value>) -> &'ll Type {
        Type::i8(cx)
    }

    pub fn char(cx: &CodegenCx<'ll, '_, &'ll Value>) -> &'ll Type {
        Type::i32(cx)
    }

    pub fn i8p(cx: &CodegenCx<'ll, '_, &'ll Value>) -> &'ll Type {
        Type::i8(cx).ptr_to()
    }

    pub fn i8p_llcx(llcx: &llvm::Context) -> &Type {
        Type::i8_llcx(llcx).ptr_to()
    }

    pub fn isize(cx: &CodegenCx<'ll, '_, &'ll Value>) -> &'ll Type {
        cx.isize_ty
    }

    pub fn c_int(cx: &CodegenCx<'ll, '_, &'ll Value>) -> &'ll Type {
        match &cx.tcx.sess.target.target.target_c_int_width[..] {
            "16" => Type::i16(cx),
            "32" => Type::i32(cx),
            "64" => Type::i64(cx),
            width => bug!("Unsupported target_c_int_width: {}", width),
        }
    }

    pub fn int_from_ty(cx: &CodegenCx<'ll, '_, &'ll Value>, t: ast::IntTy) -> &'ll Type {
        match t {
            ast::IntTy::Isize => cx.isize_ty,
            ast::IntTy::I8 => Type::i8(cx),
            ast::IntTy::I16 => Type::i16(cx),
            ast::IntTy::I32 => Type::i32(cx),
            ast::IntTy::I64 => Type::i64(cx),
            ast::IntTy::I128 => Type::i128(cx),
        }
    }

    pub fn uint_from_ty(cx: &CodegenCx<'ll, '_, &'ll Value>, t: ast::UintTy) -> &'ll Type {
        match t {
            ast::UintTy::Usize => cx.isize_ty,
            ast::UintTy::U8 => Type::i8(cx),
            ast::UintTy::U16 => Type::i16(cx),
            ast::UintTy::U32 => Type::i32(cx),
            ast::UintTy::U64 => Type::i64(cx),
            ast::UintTy::U128 => Type::i128(cx),
        }
    }

    pub fn float_from_ty(cx: &CodegenCx<'ll, '_, &'ll Value>, t: ast::FloatTy) -> &'ll Type {
        match t {
            ast::FloatTy::F32 => Type::f32(cx),
            ast::FloatTy::F64 => Type::f64(cx),
        }
    }

    pub fn func(args: &[&'ll Type], ret: &'ll Type) -> &'ll Type {
        unsafe {
            llvm::LLVMFunctionType(ret, args.as_ptr(),
                                   args.len() as c_uint, False)
        }
    }

    pub fn variadic_func(args: &[&'ll Type], ret: &'ll Type) -> &'ll Type {
        unsafe {
            llvm::LLVMFunctionType(ret, args.as_ptr(),
                                   args.len() as c_uint, True)
        }
    }

    pub fn struct_(
        cx: &CodegenCx<'ll, '_, &'ll Value>,
        els: &[&'ll Type],
        packed: bool
    ) -> &'ll Type {
        unsafe {
            llvm::LLVMStructTypeInContext(cx.llcx, els.as_ptr(),
                                          els.len() as c_uint,
                                          packed as Bool)
        }
    }

    pub fn named_struct(cx: &CodegenCx<'ll, '_, &'ll Value>, name: &str) -> &'ll Type {
        let name = SmallCStr::new(name);
        unsafe {
            llvm::LLVMStructCreateNamed(cx.llcx, name.as_ptr())
        }
    }


    pub fn array(ty: &Type, len: u64) -> &Type {
        unsafe {
            llvm::LLVMRustArrayType(ty, len)
        }
    }

    pub fn vector(ty: &Type, len: u64) -> &Type {
        unsafe {
            llvm::LLVMVectorType(ty, len as c_uint)
        }
    }

    pub fn kind(&self) -> TypeKind {
        unsafe {
            llvm::LLVMRustGetTypeKind(self)
        }
    }

    pub fn set_struct_body(&'ll self, els: &[&'ll Type], packed: bool) {
        unsafe {
            llvm::LLVMStructSetBody(self, els.as_ptr(),
                                    els.len() as c_uint, packed as Bool)
        }
    }

    pub fn ptr_to(&self) -> &Type {
        unsafe {
            llvm::LLVMPointerType(self, 0)
        }
    }

    pub fn element_type(&self) -> &Type {
        unsafe {
            llvm::LLVMGetElementType(self)
        }
    }

    /// Return the number of elements in `self` if it is a LLVM vector type.
    pub fn vector_length(&self) -> usize {
        unsafe {
            llvm::LLVMGetVectorSize(self) as usize
        }
    }

    pub fn func_params(&self) -> Vec<&Type> {
        unsafe {
            let n_args = llvm::LLVMCountParamTypes(self) as usize;
            let mut args = Vec::with_capacity(n_args);
            llvm::LLVMGetParamTypes(self, args.as_mut_ptr());
            args.set_len(n_args);
            args
        }
    }

    pub fn float_width(&self) -> usize {
        match self.kind() {
            TypeKind::Float => 32,
            TypeKind::Double => 64,
            TypeKind::X86_FP80 => 80,
            TypeKind::FP128 | TypeKind::PPC_FP128 => 128,
            _ => bug!("llvm_float_width called on a non-float type")
        }
    }

    /// Retrieve the bit width of the integer type `self`.
    pub fn int_width(&self) -> u64 {
        unsafe {
            llvm::LLVMGetIntTypeWidth(self) as u64
        }
    }

    pub fn from_integer(cx: &CodegenCx<'ll, '_, &'ll Value>, i: layout::Integer) -> &'ll Type {
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
    pub fn pointee_for_abi_align(cx: &CodegenCx<'ll, '_, &'ll Value>, align: Align) -> &'ll Type {
        // FIXME(eddyb) We could find a better approximation if ity.align < align.
        let ity = layout::Integer::approximate_abi_align(cx, align);
        Type::from_integer(cx, ity)
    }

    /// Return a LLVM type that has at most the required alignment,
    /// and exactly the required size, as a best-effort padding array.
    pub fn padding_filler(
        cx: &CodegenCx<'ll, '_, &'ll Value>,
        size: Size,
        align: Align
    ) -> &'ll Type {
        let unit = layout::Integer::approximate_abi_align(cx, align);
        let size = size.bytes();
        let unit_size = unit.size().bytes();
        assert_eq!(size % unit_size, 0);
        Type::array(Type::from_integer(cx, unit), size / unit_size)
    }

    pub fn x86_mmx(cx: &CodegenCx<'ll, '_, &'ll Value>) -> &'ll Type {
        unsafe {
            llvm::LLVMX86MMXTypeInContext(cx.llcx)
        }
    }
}
