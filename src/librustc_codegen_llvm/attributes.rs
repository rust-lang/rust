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

use rustc::hir::{self, CodegenFnAttrFlags};
use rustc::hir::def_id::{DefId, LOCAL_CRATE};
use rustc::hir::itemlikevisit::ItemLikeVisitor;
use rustc::session::Session;
use rustc::session::config::Sanitizer;
use rustc::ty::TyCtxt;
use rustc::ty::query::Providers;
use rustc_data_structures::sync::Lrc;
use rustc_data_structures::fx::FxHashMap;
use rustc_target::spec::PanicStrategy;

use attributes;
use llvm::{self, Attribute, ValueRef};
use llvm::AttributePlace::Function;
use llvm_util;
pub use syntax::attr::{self, InlineAttr};
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

    // probestack doesn't play nice either with pgo-gen.
    if cx.sess().opts.debugging_opts.pgo_gen.is_some() {
        return;
    }

    // probestack doesn't play nice either with gcov profiling.
    if cx.sess().opts.debugging_opts.profile {
        return;
    }

    // Flag our internal `__rust_probestack` function as the stack probe symbol.
    // This is defined in the `compiler-builtins` crate for each architecture.
    llvm::AddFunctionAttrStringValue(
        llfn, llvm::AttributePlace::Function,
        cstr("probe-stack\0"), cstr("__rust_probestack\0"));
}

pub fn llvm_target_features(sess: &Session) -> impl Iterator<Item = &str> {
    const RUSTC_SPECIFIC_FEATURES: &[&str] = &[
        "crt-static",
    ];

    let cmdline = sess.opts.cg.target_feature.split(',')
        .filter(|f| !RUSTC_SPECIFIC_FEATURES.iter().any(|s| f.contains(s)));
    sess.target.target.options.features.split(',')
        .chain(cmdline)
        .filter(|l| !l.is_empty())
}

/// Composite function which sets LLVM attributes for function depending on its AST (#[attribute])
/// attributes.
pub fn from_fn_attrs(cx: &CodegenCx, llfn: ValueRef, id: DefId) {
    let codegen_fn_attrs = cx.tcx.codegen_fn_attrs(id);

    inline(llfn, codegen_fn_attrs.inline);

    set_frame_pointer_elimination(cx, llfn);
    set_probestack(cx, llfn);

    if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::COLD) {
        Attribute::Cold.apply_llfn(Function, llfn);
    }
    if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::NAKED) {
        naked(llfn, true);
    }
    if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::ALLOCATOR) {
        Attribute::NoAlias.apply_llfn(
            llvm::AttributePlace::ReturnValue, llfn);
    }

    let can_unwind = if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::UNWIND) {
        Some(true)
    } else if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::RUSTC_ALLOCATOR_NOUNWIND) {
        Some(false)

    // Perhaps questionable, but we assume that anything defined
    // *in Rust code* may unwind. Foreign items like `extern "C" {
    // fn foo(); }` are assumed not to unwind **unless** they have
    // a `#[unwind]` attribute.
    } else if !cx.tcx.is_foreign_item(id) {
        Some(true)
    } else {
        None
    };

    match can_unwind {
        Some(false) => attributes::unwind(llfn, false),
        Some(true) if cx.tcx.sess.panic_strategy() == PanicStrategy::Unwind => {
            attributes::unwind(llfn, true);
        }
        Some(true) | None => {}
    }

    let features = llvm_target_features(cx.tcx.sess)
        .map(|s| s.to_string())
        .chain(
            codegen_fn_attrs.target_features
                .iter()
                .map(|f| {
                    let feature = &*f.as_str();
                    format!("+{}", llvm_util::to_llvm_feature(cx.tcx.sess, feature))
                })
        )
        .collect::<Vec<String>>()
        .join(",");

    if !features.is_empty() {
        let val = CString::new(features).unwrap();
        llvm::AddFunctionAttrStringValue(
            llfn, llvm::AttributePlace::Function,
            cstr("target-features\0"), &val);
    }

    // Note that currently the `wasm-import-module` doesn't do anything, but
    // eventually LLVM 7 should read this and ferry the appropriate import
    // module to the output file.
    if cx.tcx.sess.target.target.arch == "wasm32" {
        if let Some(module) = wasm_import_module(cx.tcx, id) {
            llvm::AddFunctionAttrStringValue(
                llfn,
                llvm::AttributePlace::Function,
                cstr("wasm-import-module\0"),
                &module,
            );
        }
    }
}

fn cstr(s: &'static str) -> &CStr {
    CStr::from_bytes_with_nul(s.as_bytes()).expect("null-terminated string")
}

pub fn provide(providers: &mut Providers) {
    providers.target_features_whitelist = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        if tcx.sess.opts.actually_rustdoc {
            // rustdoc needs to be able to document functions that use all the features, so
            // whitelist them all
            Lrc::new(llvm_util::all_known_features()
                .map(|(a, b)| (a.to_string(), b.map(|s| s.to_string())))
                .collect())
        } else {
            Lrc::new(llvm_util::target_feature_whitelist(tcx.sess)
                .iter()
                .map(|&(a, b)| (a.to_string(), b.map(|s| s.to_string())))
                .collect())
        }
    };

    providers.wasm_custom_sections = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        let mut finder = WasmSectionFinder { tcx, list: Vec::new() };
        tcx.hir.krate().visit_all_item_likes(&mut finder);
        Lrc::new(finder.list)
    };

    provide_extern(providers);
}

struct WasmSectionFinder<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    list: Vec<DefId>,
}

impl<'a, 'tcx: 'a> ItemLikeVisitor<'tcx> for WasmSectionFinder<'a, 'tcx> {
    fn visit_item(&mut self, i: &'tcx hir::Item) {
        match i.node {
            hir::ItemConst(..) => {}
            _ => return,
        }
        if i.attrs.iter().any(|i| i.check_name("wasm_custom_section")) {
            self.list.push(self.tcx.hir.local_def_id(i.id));
        }
    }

    fn visit_trait_item(&mut self, _: &'tcx hir::TraitItem) {}

    fn visit_impl_item(&mut self, _: &'tcx hir::ImplItem) {}
}

pub fn provide_extern(providers: &mut Providers) {
    providers.wasm_import_module_map = |tcx, cnum| {
        let mut ret = FxHashMap();
        for lib in tcx.foreign_modules(cnum).iter() {
            let attrs = tcx.get_attrs(lib.def_id);
            let mut module = None;
            for attr in attrs.iter().filter(|a| a.check_name("wasm_import_module")) {
                module = attr.value_str();
            }
            let module = match module {
                Some(s) => s,
                None => continue,
            };
            for id in lib.foreign_items.iter() {
                assert_eq!(id.krate, cnum);
                ret.insert(*id, module.to_string());
            }
        }

        Lrc::new(ret)
    }
}

fn wasm_import_module(tcx: TyCtxt, id: DefId) -> Option<CString> {
    tcx.wasm_import_module_map(id.krate)
        .get(&id)
        .map(|s| CString::new(&s[..]).unwrap())
}
