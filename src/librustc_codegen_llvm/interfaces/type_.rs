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
    fn void(&self) -> Self::Type;
    fn metadata(&self) -> Self::Type;
    fn i1(&self) -> Self::Type;
    fn i8(&self) -> Self::Type;
    fn i16(&self) -> Self::Type;
    fn i32(&self) -> Self::Type;
    fn i64(&self) -> Self::Type;
    fn i128(&self) -> Self::Type;
    fn ix(&self, num_bites: u64) -> Self::Type;
    fn f32(&self) -> Self::Type;
    fn f64(&self) -> Self::Type;
    fn bool(&self) -> Self::Type;
    fn char(&self) -> Self::Type;
    fn i8p(&self) -> Self::Type;

    fn func(&self, args: &[Self::Type], ret: Self::Type) -> Self::Type;
    fn variadic_func(&self, args: &[Self::Type]) -> Self::Type;
    fn struct_(&self, els: &[Self::Type], packed: bool) -> Self::Type;
    fn named_struct(&self, name: &str) -> Self::Type;
    fn array(&self, ty: Self::Type, len: u64) -> Self::Type;
    fn vector(&self, ty: Self::Type, len: u64) -> Self::Type;
    fn kind(&self, ty: Self::Type) -> Self::TypeKind;
    fn set_struct_body(&self, els: &[Self::Type], packed: bool);
    fn ptr_to(&self, ty: Self::Type) -> Self::Type;
    fn element_type(&self, ty: Self::Type) -> Self::Type;
    fn vector_length(&self, ty: Self::Type) -> usize;
    fn func_params(&self, ty: Self::Type) -> Vec<Self::Type>;
    fn float_width(&self, ty: Self::Type) -> usize;
    fn int_width(&self, ty: Self::Type) -> usize;
}
