// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::backend::Backend;

pub trait TypeMethods : Backend {
    fn type_void(&self) -> Self::Type;
    fn type_metadata(&self) -> Self::Type;
    fn type_i1(&self) -> Self::Type;
    fn type_i8(&self) -> Self::Type;
    fn type_i16(&self) -> Self::Type;
    fn type_i32(&self) -> Self::Type;
    fn type_i64(&self) -> Self::Type;
    fn type_i128(&self) -> Self::Type;
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
    fn type_kind(&self, ty: Self::Type) -> Self::TypeKind;
    fn set_struct_body(&self, ty: Self::Type, els: &[Self::Type], packed: bool);
    fn type_ptr_to(&self, ty: Self::Type) -> Self::Type;
    fn element_type(&self, ty: Self::Type) -> Self::Type;
    fn vector_length(&self, ty: Self::Type) -> usize;
    fn func_params_types(&self, ty: Self::Type) -> Vec<Self::Type>;
    fn float_width(&self, ty: Self::Type) -> usize;
    fn int_width(&self, ty: Self::Type) -> u64;

    fn val_ty(&self, v: Self::Value) -> Self::Type;
}
