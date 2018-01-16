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
use std::rc::Rc;

use rustc::hir::Unsafety;
use rustc::hir::def_id::{DefId, LOCAL_CRATE};
use rustc::session::config::Sanitizer;
use rustc::ty::TyCtxt;
use rustc::ty::maps::Providers;
use rustc_data_structures::fx::FxHashSet;

use llvm::{self, Attribute, ValueRef};
use llvm::AttributePlace::Function;
use llvm_util;
pub use syntax::attr::{self, InlineAttr};
use syntax::ast;
use context::CodegenCx;

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

/// Tell LLVM whether it should optimize function for size.
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

pub fn set_frame_pointer_elimination(cx: &CodegenCx, llfn: ValueRef) {
    // FIXME: #11906: Omitting frame pointers breaks retrieving the value of a
    // parameter.
    if cx.sess().must_not_eliminate_frame_pointers() {
        llvm::AddFunctionAttrStringValue(
            llfn, llvm::AttributePlace::Function,
            cstr("no-frame-pointer-elim\0"), cstr("true\0"));
    }
}

pub fn set_probestack(cx: &CodegenCx, llfn: ValueRef) {
    // Only use stack probes if the target specification indicates that we
    // should be using stack probes
    if !cx.sess().target.target.options.stack_probes {
        return
    }

    // Currently stack probes seem somewhat incompatible with the address
    // sanitizer. With asan we're already protected from stack overflow anyway
    // so we don't really need stack probes regardless.
    match cx.sess().opts.debugging_opts.sanitizer {
        Some(Sanitizer::Address) => return,
        _ => {}
    }

    // Flag our internal `__rust_probestack` function as the stack probe symbol.
    // This is defined in the `compiler-builtins` crate for each architecture.
    llvm::AddFunctionAttrStringValue(
        llfn, llvm::AttributePlace::Function,
        cstr("probe-stack\0"), cstr("__rust_probestack\0"));
}

/// Composite function which sets LLVM attributes for function depending on its AST (#[attribute])
/// attributes.
pub fn from_fn_attrs(cx: &CodegenCx, llfn: ValueRef, id: DefId) {
    use syntax::attr::*;
    let attrs = cx.tcx.get_attrs(id);
    inline(llfn, find_inline_attr(Some(cx.sess().diagnostic()), &attrs));

    set_frame_pointer_elimination(cx, llfn);
    set_probestack(cx, llfn);

    for attr in attrs.iter() {
        if attr.check_name("cold") {
            Attribute::Cold.apply_llfn(Function, llfn);
        } else if attr.check_name("naked") {
            naked(llfn, true);
        } else if attr.check_name("allocator") {
            Attribute::NoAlias.apply_llfn(
                llvm::AttributePlace::ReturnValue, llfn);
        } else if attr.check_name("unwind") {
            unwind(llfn, true);
        } else if attr.check_name("rustc_allocator_nounwind") {
            unwind(llfn, false);
        }
    }

    let target_features = cx.tcx.target_features_enabled(id);
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

pub fn provide(providers: &mut Providers) {
    providers.target_features_whitelist = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        Rc::new(llvm_util::target_feature_whitelist(tcx.sess)
            .iter()
            .map(|c| c.to_str().unwrap().to_string())
            .collect())
    };

    providers.target_features_enabled = |tcx, id| {
        let whitelist = tcx.target_features_whitelist(LOCAL_CRATE);
        let mut target_features = Vec::new();
        for attr in tcx.get_attrs(id).iter() {
            if !attr.check_name("target_feature") {
                continue
            }
            if let Some(val) = attr.value_str() {
                for feat in val.as_str().split(",").map(|f| f.trim()) {
                    if !feat.is_empty() && !feat.contains('\0') {
                        target_features.push(feat.to_string());
                    }
                }
                let msg = "#[target_feature = \"..\"] is deprecated and will \
                           eventually be removed, use \
                           #[target_feature(enable = \"..\")] instead";
                tcx.sess.span_warn(attr.span, &msg);
                continue
            }

            if tcx.fn_sig(id).unsafety() == Unsafety::Normal {
                let msg = "#[target_feature(..)] can only be applied to \
                           `unsafe` function";
                tcx.sess.span_err(attr.span, msg);
            }
            from_target_feature(tcx, attr, &whitelist, &mut target_features);
        }
        Rc::new(target_features)
    };
}

fn from_target_feature(
    tcx: TyCtxt,
    attr: &ast::Attribute,
    whitelist: &FxHashSet<String>,
    target_features: &mut Vec<String>,
) {
    let list = match attr.meta_item_list() {
        Some(list) => list,
        None => {
            let msg = "#[target_feature] attribute must be of the form \
                       #[target_feature(..)]";
            tcx.sess.span_err(attr.span, &msg);
            return
        }
    };

    for item in list {
        if !item.check_name("enable") {
            let msg = "#[target_feature(..)] only accepts sub-keys of `enable` \
                       currently";
            tcx.sess.span_err(item.span, &msg);
            continue
        }
        let value = match item.value_str() {
            Some(list) => list,
            None => {
                let msg = "#[target_feature] attribute must be of the form \
                           #[target_feature(enable = \"..\")]";
                tcx.sess.span_err(item.span, &msg);
                continue
            }
        };
        let value = value.as_str();
        for feature in value.split(',') {
            if whitelist.contains(feature) {
                target_features.push(format!("+{}", feature));
                continue
            }

            let msg = format!("the feature named `{}` is not valid for \
                               this target", feature);
            let mut err = tcx.sess.struct_span_err(item.span, &msg);

            if feature.starts_with("+") {
                let valid = whitelist.contains(&feature[1..]);
                if valid {
                    err.help("consider removing the leading `+` in the feature name");
                }
            }
            err.emit();
        }
    }
}
