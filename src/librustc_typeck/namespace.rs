// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::hir;
use rustc::ty;

// Whether an item exists in the type or value namespace.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Namespace {
    Type,
    Value,
}

impl From<ty::AssociatedKind> for Namespace {
    fn from(a_kind: ty::AssociatedKind) -> Self {
        match a_kind {
            ty::AssociatedKind::Type => Namespace::Type,
            ty::AssociatedKind::Const |
            ty::AssociatedKind::Method => Namespace::Value,
        }
    }
}

impl<'a> From <&'a hir::ImplItemKind> for Namespace {
    fn from(impl_kind: &'a hir::ImplItemKind) -> Self {
        match *impl_kind {
            hir::ImplItemKind::Type(..) => Namespace::Type,
            hir::ImplItemKind::Const(..) |
            hir::ImplItemKind::Method(..) => Namespace::Value,
        }
    }
}
