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

use std::ffi::CString;

use rustc::hir::{CodegenFnAttrFlags, CodegenFnAttrs};
use rustc::hir::def_id::{DefId, LOCAL_CRATE};
use rustc::session::Session;
use rustc::session::config::Sanitizer;
use rustc::ty::TyCtxt;
use rustc::ty::query::Providers;
use rustc_data_structures::sync::Lrc;
use rustc_data_structures::fx::FxHashMap;
use rustc_target::spec::PanicStrategy;

use attributes;
use llvm::{self, Attribute};
use llvm::AttributePlace::Function;
use llvm_util;
pub use syntax::attr::{self, InlineAttr};

use context::CodegenCx;
use value::Value;

/// Mark LLVM function to use provided inline heuristic.
#[inline]
pub fn inline(val: &'ll Value, inline: InlineAttr) {
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
pub fn emit_uwtable(val: &'ll Value, emit: bool) {
    Attribute::UWTable.toggle_llfn(Function, val, emit);
}

/// Tell LLVM whether the function can or cannot unwind.
#[inline]
pub fn unwind(val: &'ll Value, can_unwind: bool) {
    Attribute::NoUnwind.toggle_llfn(Function, val, !can_unwind);
}

/// Tell LLVM whether it should optimize function for size.
#[inline]
#[allow(dead_code)] // possibly useful function
pub fn set_optimize_for_size(val: &'ll Value, optimize: bool) {
    Attribute::OptimizeForSize.toggle_llfn(Function, val, optimize);
}

/// Tell LLVM if this function should be 'naked', i.e. skip the epilogue and prologue.
#[inline]
pub fn naked(val: &'ll Value, is_naked: bool) {
    Attribute::Naked.toggle_llfn(Function, val, is_naked);
}

pub fn set_frame_pointer_elimination(cx: &CodegenCx<'ll, '_>, llfn: &'ll Value) {
    if cx.sess().must_not_eliminate_frame_pointers() {
        llvm::AddFunctionAttrStringValue(
            llfn, llvm::AttributePlace::Function,
            const_cstr!("no-frame-pointer-elim"), const_cstr!("true"));
    }
}

pub fn set_probestack(cx: &CodegenCx<'ll, '_>, llfn: &'ll Value) {
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
        const_cstr!("probe-stack"), const_cstr!("__rust_probestack"));
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

pub fn apply_target_cpu_attr(cx: &CodegenCx<'ll, '_>, llfn: &'ll Value) {
    let cpu = llvm_util::target_cpu(cx.tcx.sess);
    let target_cpu = CString::new(cpu).unwrap();
    llvm::AddFunctionAttrStringValue(
            llfn,
            llvm::AttributePlace::Function,
            const_cstr!("target-cpu"),
            target_cpu.as_c_str());
}

/// Composite function which sets LLVM attributes for function depending on its AST (#[attribute])
/// attributes.
pub fn from_fn_attrs(
    cx: &CodegenCx<'ll, '_>,
    llfn: &'ll Value,
    id: Option<DefId>,
) {
    let codegen_fn_attrs = id.map(|id| cx.tcx.codegen_fn_attrs(id))
        .unwrap_or(CodegenFnAttrs::new());

    inline(llfn, codegen_fn_attrs.inline);

    // The `uwtable` attribute according to LLVM is:
    //
    //     This attribute indicates that the ABI being targeted requires that an
    //     unwind table entry be produced for this function even if we can show
    //     that no exceptions passes by it. This is normally the case for the
    //     ELF x86-64 abi, but it can be disabled for some compilation units.
    //
    // Typically when we're compiling with `-C panic=abort` (which implies this
    // `no_landing_pads` check) we don't need `uwtable` because we can't
    // generate any exceptions! On Windows, however, exceptions include other
    // events such as illegal instructions, segfaults, etc. This means that on
    // Windows we end up still needing the `uwtable` attribute even if the `-C
    // panic=abort` flag is passed.
    //
    // You can also find more info on why Windows is whitelisted here in:
    //      https://bugzilla.mozilla.org/show_bug.cgi?id=1302078
    if !cx.sess().no_landing_pads() ||
       cx.sess().target.target.options.requires_uwtable {
        attributes::emit_uwtable(llfn, true);
    }

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
    } else if id.map(|id| !cx.tcx.is_foreign_item(id)).unwrap_or(false) {
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

    // Always annotate functions with the target-cpu they are compiled for.
    // Without this, ThinLTO won't inline Rust functions into Clang generated
    // functions (because Clang annotates functions this way too).
    // NOTE: For now we just apply this if -Zcross-lang-lto is specified, since
    //       it introduce a little overhead and isn't really necessary otherwise.
    if cx.tcx.sess.opts.debugging_opts.cross_lang_lto.enabled() {
        apply_target_cpu_attr(cx, llfn);
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
            const_cstr!("target-features"), &val);
    }

    // Note that currently the `wasm-import-module` doesn't do anything, but
    // eventually LLVM 7 should read this and ferry the appropriate import
    // module to the output file.
    if let Some(id) = id {
        if cx.tcx.sess.target.target.arch == "wasm32" {
            if let Some(module) = wasm_import_module(cx.tcx, id) {
                llvm::AddFunctionAttrStringValue(
                    llfn,
                    llvm::AttributePlace::Function,
                    const_cstr!("wasm-import-module"),
                    &module,
                );
            }
        }
    }
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

    provide_extern(providers);
}

pub fn provide_extern(providers: &mut Providers) {
    providers.wasm_import_module_map = |tcx, cnum| {
        // Build up a map from DefId to a `NativeLibrary` structure, where
        // `NativeLibrary` internally contains information about
        // `#[link(wasm_import_module = "...")]` for example.
        let native_libs = tcx.native_libraries(cnum);
        let mut def_id_to_native_lib = FxHashMap();
        for lib in native_libs.iter() {
            if let Some(id) = lib.foreign_module {
                def_id_to_native_lib.insert(id, lib);
            }
        }

        let mut ret = FxHashMap();
        for lib in tcx.foreign_modules(cnum).iter() {
            let module = def_id_to_native_lib
                .get(&lib.def_id)
                .and_then(|s| s.wasm_import_module);
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
    };
}

fn wasm_import_module(tcx: TyCtxt, id: DefId) -> Option<CString> {
    tcx.wasm_import_module_map(id.krate)
        .get(&id)
        .map(|s| CString::new(&s[..]).unwrap())
}
