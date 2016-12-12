// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//! Set and unset common attributes on LLVM values.

use std::ffi::{CStr, CString};

use llvm::{self, Attribute, ValueRef};
use llvm::AttributePlace::Function;
pub use syntax::attr::InlineAttr;
use syntax::ast;
use context::CrateContext;

/// Mark LLVM function to use provided inline heuristic.
#[inline]
pub fn inline(val: ValueRef, inline: InlineAttr) {
    use self::InlineAttr::*;
    match inline {
        Hint   => Attribute::InlineHint.apply_llfn(Function, val),
        Always => Attribute::AlwaysInline.apply_llfn(Function, val),
        Never  => Attribute::NoInline.apply_llfn(Function, val),
        None   => {
            Attribute::InlineHint.unapply_llfn(Function, val);
            Attribute::AlwaysInline.unapply_llfn(Function, val);
            Attribute::NoInline.unapply_llfn(Function, val);
        },
    };
}

/// Tell LLVM to emit or not emit the information necessary to unwind the stack for the function.
#[inline]
pub fn emit_uwtable(val: ValueRef, emit: bool) {
    Attribute::UWTable.toggle_llfn(Function, val, emit);
}

/// Tell LLVM whether the function can or cannot unwind.
#[inline]
pub fn unwind(val: ValueRef, can_unwind: bool) {
    Attribute::NoUnwind.toggle_llfn(Function, val, !can_unwind);
}

/// Tell LLVM whether it should optimise function for size.
#[inline]
#[allow(dead_code)] // possibly useful function
pub fn set_optimize_for_size(val: ValueRef, optimize: bool) {
    Attribute::OptimizeForSize.toggle_llfn(Function, val, optimize);
}

/// Tell LLVM if this function should be 'naked', i.e. skip the epilogue and prologue.
#[inline]
pub fn naked(val: ValueRef, is_naked: bool) {
    Attribute::Naked.toggle_llfn(Function, val, is_naked);
}

pub fn set_frame_pointer_elimination(ccx: &CrateContext, llfn: ValueRef) {
    // FIXME: #11906: Omitting frame pointers breaks retrieving the value of a
    // parameter.
    if ccx.sess().must_not_eliminate_frame_pointers() {
        llvm::AddFunctionAttrStringValue(
            llfn, llvm::AttributePlace::Function,
            cstr("no-frame-pointer-elim\0"), cstr("true\0"));
    }
}

/// Composite function which sets LLVM attributes for function depending on its AST (#[attribute])
/// attributes.
pub fn from_fn_attrs(ccx: &CrateContext, attrs: &[ast::Attribute], llfn: ValueRef) {
    use syntax::attr::*;
    inline(llfn, find_inline_attr(Some(ccx.sess().diagnostic()), attrs));

    set_frame_pointer_elimination(ccx, llfn);
    let mut target_features = vec![];
    for attr in attrs {
        if attr.check_name("target_feature") {
            if let Some(val) = attr.value_str() {
                for feat in val.as_str().split(",").map(|f| f.trim()) {
                    if !feat.is_empty() && !feat.contains('\0') {
                        target_features.push(feat.to_string());
                    }
                }
            }
        } else if attr.check_name("cold") {
            Attribute::Cold.apply_llfn(Function, llfn);
        } else if attr.check_name("naked") {
            naked(llfn, true);
        } else if attr.check_name("allocator") {
            Attribute::NoAlias.apply_llfn(
                llvm::AttributePlace::ReturnValue(), llfn);
        } else if attr.check_name("unwind") {
            unwind(llfn, true);
        }
    }
    if !target_features.is_empty() {
        let val = CString::new(target_features.join(",")).unwrap();
        llvm::AddFunctionAttrStringValue(
            llfn, llvm::AttributePlace::Function,
            cstr("target-features\0"), &val);
    }
}

fn cstr(s: &'static str) -> &CStr {
    CStr::from_bytes_with_nul(s.as_bytes()).expect("null-terminated string")
}
