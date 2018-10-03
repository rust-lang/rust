// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::Backend;
use super::builder::HasCodegen;
use common::TypeKind;
use syntax::ast;
use rustc::ty::layout::{self, Align, Size};
use std::cell::RefCell;
use rustc::util::nodemap::FxHashMap;
use rustc::ty::{Ty, TyCtxt};
use rustc::ty::layout::TyLayout;
use rustc_target::abi::call::{ArgType, CastTarget, FnType, Reg};
use mir::place::PlaceRef;


pub trait BaseTypeMethods<'ll, 'tcx: 'll> : Backend<'ll> {
    fn type_void(&self) -> Self::Type;
    fn type_metadata(&self) -> Self::Type;
    fn type_i1(&self) -> Self::Type;
    fn type_i8(&self) -> Self::Type;
    fn type_i16(&self) -> Self::Type;
    fn type_i32(&self) -> Self::Type;
    fn type_i64(&self) -> Self::Type;
    fn type_i128(&self) -> Self::Type;

    // Creates an integer type with the given number of bits, e.g. i24
    fn type_ix(&self, num_bits: u64) -> Self::Type;

    fn type_f32(&self) -> Self::Type;
    fn type_f64(&self) -> Self::Type;
    fn type_x86_mmx(&self) -> Self::Type;

    fn type_func(&self, args: &[Self::Type], ret: Self::Type) -> Self::Type;
    fn type_variadic_func(&self, args: &[Self::Type], ret: Self::Type) -> Self::Type;
    fn type_struct(&self, els: &[Self::Type], packed: bool) -> Self::Type;
    fn type_named_struct(&self, name: &str) -> Self::Type;
    fn type_array(&self, ty: Self::Type, len: u64) -> Self::Type;
    fn type_vector(&self, ty: Self::Type, len: u64) -> Self::Type;
    fn type_kind(&self, ty: Self::Type) -> TypeKind;
    fn set_struct_body(&self, ty: Self::Type, els: &[Self::Type], packed: bool);
    fn type_ptr_to(&self, ty: Self::Type) -> Self::Type;
    fn element_type(&self, ty: Self::Type) -> Self::Type;

    /// Return the number of elements in `self` if it is a LLVM vector type.
    fn vector_length(&self, ty: Self::Type) -> usize;

    fn func_params_types(&self, ty: Self::Type) -> Vec<Self::Type>;
    fn float_width(&self, ty: Self::Type) -> usize;

    /// Retrieve the bit width of the integer type `self`.
    fn int_width(&self, ty: Self::Type) -> u64;

    fn val_ty(&self, v: Self::Value) -> Self::Type;
    fn scalar_lltypes(&self) -> &RefCell<FxHashMap<Ty<'tcx>, Self::Type>>;
    fn tcx(&self) -> &TyCtxt<'ll, 'tcx, 'tcx>;
}

pub trait DerivedTypeMethods<'ll, 'tcx: 'll> : Backend<'ll> {
    fn type_bool(&self) -> Self::Type;
    fn type_char(&self) -> Self::Type;
    fn type_i8p(&self) -> Self::Type;
    fn type_isize(&self) -> Self::Type;
    fn type_int(&self) -> Self::Type;
    fn type_int_from_ty(
        &self,
        t: ast::IntTy
    ) -> Self::Type;
    fn type_uint_from_ty(
        &self,
        t: ast::UintTy
    ) -> Self::Type;
    fn type_float_from_ty(
        &self,
        t: ast::FloatTy
    ) -> Self::Type;
    fn type_from_integer(&self, i: layout::Integer) -> Self::Type;

    /// Return a LLVM type that has at most the required alignment,
    /// as a conservative approximation for unknown pointee types.
    fn type_pointee_for_abi_align(&self, align: Align) -> Self::Type;

    /// Return a LLVM type that has at most the required alignment,
    /// and exactly the required size, as a best-effort padding array.
    fn type_padding_filler(
        &self,
        size: Size,
        align: Align
    ) -> Self::Type;

    fn type_needs_drop(&self, ty: Ty<'tcx>) -> bool;
    fn type_is_sized(&self, ty: Ty<'tcx>) -> bool;
    fn type_is_freeze(&self, ty: Ty<'tcx>) -> bool;
    fn type_has_metadata(&self, ty: Ty<'tcx>) -> bool;
}

pub trait LayoutTypeMethods<'ll, 'tcx> : Backend<'ll> {
    fn backend_type(&self, ty: &TyLayout<'tcx>) -> Self::Type;
    fn cast_backend_type(&self, ty: &CastTarget) -> Self::Type;
    fn fn_backend_type(&self, ty: &FnType<'tcx, Ty<'tcx>>) -> Self::Type;
    fn reg_backend_type(&self, ty: &Reg) -> Self::Type;
    fn immediate_backend_type(&self, ty: &TyLayout<'tcx>) -> Self::Type;
    fn is_backend_immediate(&self, ty: &TyLayout<'tcx>) -> bool;
    fn is_backend_scalar_pair(&self, ty: &TyLayout<'tcx>) -> bool;
    fn backend_field_index(&self, ty: &TyLayout<'tcx>, index: usize) -> u64;
    fn scalar_pair_element_backend_type<'a>(
        &self,
        ty: &TyLayout<'tcx>,
        index: usize,
        immediate: bool
    ) -> Self::Type;
}

pub trait ArgTypeMethods<'a, 'll: 'a, 'tcx: 'll> : HasCodegen<'a, 'll, 'tcx> {
    fn store_fn_arg(
        &self,
        ty: &ArgType<'tcx, Ty<'tcx>>,
        idx: &mut usize, dst: PlaceRef<'tcx, <Self::CodegenCx as Backend<'ll>>::Value>
    );
    fn store_arg_ty(
        &self,
        ty: &ArgType<'tcx, Ty<'tcx>>,
        val: <Self::CodegenCx as Backend<'ll>>::Value,
        dst: PlaceRef<'tcx, <Self::CodegenCx as Backend<'ll>>::Value>
    );
    fn memory_ty(&self, ty: &ArgType<'tcx, Ty<'tcx>>) -> <Self::CodegenCx as Backend<'ll>>::Type;
}

pub trait TypeMethods<'ll, 'tcx: 'll> :
    BaseTypeMethods<'ll, 'tcx> + DerivedTypeMethods<'ll, 'tcx> + LayoutTypeMethods<'ll, 'tcx> {}
