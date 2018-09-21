// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::ty::Ty;
use super::backend::Backend;
use rustc::hir::def_id::DefId;
use rustc::mir::mono::{Linkage, Visibility};
use monomorphize::Instance;

pub trait DeclareMethods<'ll, 'tcx: 'll> : Backend<'ll> {
    fn declare_global(
        &self,
        name: &str, ty: Self::Type
    ) -> Self::Value;
    fn declare_cfn(
        &self,
        name: &str,
        fn_type: Self::Type
    ) -> Self::Value;
    fn declare_fn(
        &self,
        name: &str,
        fn_type: Ty<'tcx>,
    ) -> Self::Value;
    fn define_global(
        &self,
        name: &str,
        ty: Self::Type
    ) -> Option<Self::Value>;
    fn define_private_global(&self, ty: Self::Type) -> Self::Value;
    fn define_fn(
        &self,
        name: &str,
        fn_type: Ty<'tcx>,
    ) -> Self::Value;
    fn define_internal_fn(
        &self,
        name: &str,
        fn_type: Ty<'tcx>,
    ) -> Self::Value;
    fn get_declared_value(&self, name: &str) -> Option<Self::Value>;
    fn get_defined_value(&self, name: &str) -> Option<Self::Value>;
}

pub trait PreDefineMethods<'ll, 'tcx: 'll> : Backend<'ll> {
    fn predefine_static(
        &self,
        def_id: DefId,
        linkage: Linkage,
        visibility: Visibility,
        symbol_name: &str
    );
    fn predefine_fn(
        &self,
        instance: Instance<'tcx>,
        linkage: Linkage,
        visibility: Visibility,
        symbol_name: &str
    );
}
