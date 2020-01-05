//! Set and unset common attributes on LLVM values.

use std::ffi::CString;

use rustc::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc::session::config::{OptLevel, Sanitizer};
use rustc::session::Session;
use rustc::ty::layout::HasTyCtxt;
use rustc::ty::query::Providers;
use rustc::ty::{self, Ty, TyCtxt};
use rustc_codegen_ssa::traits::*;
use rustc_data_structures::const_cstr;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::small_c_str::SmallCStr;
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_target::abi::call::Conv;
use rustc_target::spec::PanicStrategy;

use crate::abi::FnAbi;
use crate::attributes;
use crate::llvm::AttributePlace::Function;
use crate::llvm::{self, Attribute};
use crate::llvm_util;
pub use syntax::attr::{self, InlineAttr, OptimizeAttr};

use crate::context::CodegenCx;
use crate::value::Value;

/// Mark LLVM function to use provided inline heuristic.
#[inline]
fn inline(cx: &CodegenCx<'ll, '_>, val: &'ll Value, inline: InlineAttr) {
    use self::InlineAttr::*;
    match inline {
        Hint => Attribute::InlineHint.apply_llfn(Function, val),
        Always => Attribute::AlwaysInline.apply_llfn(Function, val),
        Never => {
            if cx.tcx().sess.target.target.arch != "amdgpu" {
                Attribute::NoInline.apply_llfn(Function, val);
            }
        }
        None => {
            Attribute::InlineHint.unapply_llfn(Function, val);
            Attribute::AlwaysInline.unapply_llfn(Function, val);
            Attribute::NoInline.unapply_llfn(Function, val);
        }
    };
}

/// Tell LLVM to emit or not emit the information necessary to unwind the stack for the function.
#[inline]
pub fn emit_uwtable(val: &'ll Value, emit: bool) {
    Attribute::UWTable.toggle_llfn(Function, val, emit);
}

/// Tell LLVM whether the function can or cannot unwind.
#[inline]
fn unwind(val: &'ll Value, can_unwind: bool) {
    Attribute::NoUnwind.toggle_llfn(Function, val, !can_unwind);
}

/// Tell LLVM if this function should be 'naked', i.e., skip the epilogue and prologue.
#[inline]
fn naked(val: &'ll Value, is_naked: bool) {
    Attribute::Naked.toggle_llfn(Function, val, is_naked);
}

pub fn set_frame_pointer_elimination(cx: &CodegenCx<'ll, '_>, llfn: &'ll Value) {
    if cx.sess().must_not_eliminate_frame_pointers() {
        if llvm_util::get_major_version() >= 8 {
            llvm::AddFunctionAttrStringValue(
                llfn,
                llvm::AttributePlace::Function,
                const_cstr!("frame-pointer"),
                const_cstr!("all"),
            );
        } else {
            llvm::AddFunctionAttrStringValue(
                llfn,
                llvm::AttributePlace::Function,
                const_cstr!("no-frame-pointer-elim"),
                const_cstr!("true"),
            );
        }
    }
}

/// Tell LLVM what instrument function to insert.
#[inline]
fn set_instrument_function(cx: &CodegenCx<'ll, '_>, llfn: &'ll Value) {
    if cx.sess().instrument_mcount() {
        // Similar to `clang -pg` behavior. Handled by the
        // `post-inline-ee-instrument` LLVM pass.

        // The function name varies on platforms.
        // See test/CodeGen/mcount.c in clang.
        let mcount_name =
            CString::new(cx.sess().target.target.options.target_mcount.as_str().as_bytes())
                .unwrap();

        llvm::AddFunctionAttrStringValue(
            llfn,
            llvm::AttributePlace::Function,
            const_cstr!("instrument-function-entry-inlined"),
            &mcount_name,
        );
    }
}

fn set_probestack(cx: &CodegenCx<'ll, '_>, llfn: &'ll Value) {
    // Only use stack probes if the target specification indicates that we
    // should be using stack probes
    if !cx.sess().target.target.options.stack_probes {
        return;
    }

    // Currently stack probes seem somewhat incompatible with the address
    // sanitizer and thread sanitizer. With asan we're already protected from
    // stack overflow anyway so we don't really need stack probes regardless.
    match cx.sess().opts.debugging_opts.sanitizer {
        Some(Sanitizer::Address) | Some(Sanitizer::Thread) => return,
        _ => {}
    }

    // probestack doesn't play nice either with `-C profile-generate`.
    if cx.sess().opts.cg.profile_generate.enabled() {
        return;
    }

    // probestack doesn't play nice either with gcov profiling.
    if cx.sess().opts.debugging_opts.profile {
        return;
    }

    // Flag our internal `__rust_probestack` function as the stack probe symbol.
    // This is defined in the `compiler-builtins` crate for each architecture.
    llvm::AddFunctionAttrStringValue(
        llfn,
        llvm::AttributePlace::Function,
        const_cstr!("probe-stack"),
        const_cstr!("__rust_probestack"),
    );
}

fn translate_obsolete_target_features(feature: &str) -> &str {
    const LLVM9_FEATURE_CHANGES: &[(&str, &str)] =
        &[("+fp-only-sp", "-fp64"), ("-fp-only-sp", "+fp64"), ("+d16", "-d32"), ("-d16", "+d32")];
    if llvm_util::get_major_version() >= 9 {
        for &(old, new) in LLVM9_FEATURE_CHANGES {
            if feature == old {
                return new;
            }
        }
    } else {
        for &(old, new) in LLVM9_FEATURE_CHANGES {
            if feature == new {
                return old;
            }
        }
    }
    feature
}

pub fn llvm_target_features(sess: &Session) -> impl Iterator<Item = &str> {
    const RUSTC_SPECIFIC_FEATURES: &[&str] = &["crt-static"];

    let cmdline = sess
        .opts
        .cg
        .target_feature
        .split(',')
        .filter(|f| !RUSTC_SPECIFIC_FEATURES.iter().any(|s| f.contains(s)));
    sess.target
        .target
        .options
        .features
        .split(',')
        .chain(cmdline)
        .filter(|l| !l.is_empty())
        .map(translate_obsolete_target_features)
}

pub fn apply_target_cpu_attr(cx: &CodegenCx<'ll, '_>, llfn: &'ll Value) {
    let target_cpu = SmallCStr::new(llvm_util::target_cpu(cx.tcx.sess));
    llvm::AddFunctionAttrStringValue(
        llfn,
        llvm::AttributePlace::Function,
        const_cstr!("target-cpu"),
        target_cpu.as_c_str(),
    );
}

/// Sets the `NonLazyBind` LLVM attribute on a given function,
/// assuming the codegen options allow skipping the PLT.
pub fn non_lazy_bind(sess: &Session, llfn: &'ll Value) {
    // Don't generate calls through PLT if it's not necessary
    if !sess.needs_plt() {
        Attribute::NonLazyBind.apply_llfn(Function, llfn);
    }
}

pub(crate) fn default_optimisation_attrs(sess: &Session, llfn: &'ll Value) {
    match sess.opts.optimize {
        OptLevel::Size => {
            llvm::Attribute::MinSize.unapply_llfn(Function, llfn);
            llvm::Attribute::OptimizeForSize.apply_llfn(Function, llfn);
            llvm::Attribute::OptimizeNone.unapply_llfn(Function, llfn);
        }
        OptLevel::SizeMin => {
            llvm::Attribute::MinSize.apply_llfn(Function, llfn);
            llvm::Attribute::OptimizeForSize.apply_llfn(Function, llfn);
            llvm::Attribute::OptimizeNone.unapply_llfn(Function, llfn);
        }
        OptLevel::No => {
            llvm::Attribute::MinSize.unapply_llfn(Function, llfn);
            llvm::Attribute::OptimizeForSize.unapply_llfn(Function, llfn);
            llvm::Attribute::OptimizeNone.unapply_llfn(Function, llfn);
        }
        _ => {}
    }
}

/// Composite function which sets LLVM attributes for function depending on its AST (`#[attribute]`)
/// attributes.
pub fn from_fn_attrs(
    cx: &CodegenCx<'ll, 'tcx>,
    llfn: &'ll Value,
    instance: ty::Instance<'tcx>,
    fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
) {
    let codegen_fn_attrs = cx.tcx.codegen_fn_attrs(instance.def_id());

    match codegen_fn_attrs.optimize {
        OptimizeAttr::None => {
            default_optimisation_attrs(cx.tcx.sess, llfn);
        }
        OptimizeAttr::Speed => {
            llvm::Attribute::MinSize.unapply_llfn(Function, llfn);
            llvm::Attribute::OptimizeForSize.unapply_llfn(Function, llfn);
            llvm::Attribute::OptimizeNone.unapply_llfn(Function, llfn);
        }
        OptimizeAttr::Size => {
            llvm::Attribute::MinSize.apply_llfn(Function, llfn);
            llvm::Attribute::OptimizeForSize.apply_llfn(Function, llfn);
            llvm::Attribute::OptimizeNone.unapply_llfn(Function, llfn);
        }
    }

    // FIXME(eddyb) consolidate these two `inline` calls (and avoid overwrites).
    if instance.def.is_inline(cx.tcx) {
        inline(cx, llfn, attributes::InlineAttr::Hint);
    }

    inline(cx, llfn, codegen_fn_attrs.inline);

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
    if !cx.sess().no_landing_pads() || cx.sess().target.target.options.requires_uwtable {
        attributes::emit_uwtable(llfn, true);
    }

    set_frame_pointer_elimination(cx, llfn);
    set_instrument_function(cx, llfn);
    set_probestack(cx, llfn);

    if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::COLD) {
        Attribute::Cold.apply_llfn(Function, llfn);
    }
    if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::FFI_RETURNS_TWICE) {
        Attribute::ReturnsTwice.apply_llfn(Function, llfn);
    }
    if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::NAKED) {
        naked(llfn, true);
    }
    if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::ALLOCATOR) {
        Attribute::NoAlias.apply_llfn(llvm::AttributePlace::ReturnValue, llfn);
    }

    unwind(
        llfn,
        if cx.tcx.sess.panic_strategy() != PanicStrategy::Unwind {
            // In panic=abort mode we assume nothing can unwind anywhere, so
            // optimize based on this!
            false
        } else if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::UNWIND) {
            // If a specific #[unwind] attribute is present, use that.
            true
        } else if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::RUSTC_ALLOCATOR_NOUNWIND) {
            // Special attribute for allocator functions, which can't unwind.
            false
        } else {
            if fn_abi.conv == Conv::Rust {
                // Any Rust method (or `extern "Rust" fn` or `extern
                // "rust-call" fn`) is explicitly allowed to unwind
                // (unless it has no-unwind attribute, handled above).
                true
            } else {
                // Anything else is either:
                //
                //  1. A foreign item using a non-Rust ABI (like `extern "C" { fn foo(); }`), or
                //
                //  2. A Rust item using a non-Rust ABI (like `extern "C" fn foo() { ... }`).
                //
                // Foreign items (case 1) are assumed to not unwind; it is
                // UB otherwise. (At least for now; see also
                // rust-lang/rust#63909 and Rust RFC 2753.)
                //
                // Items defined in Rust with non-Rust ABIs (case 2) are also
                // not supposed to unwind. Whether this should be enforced
                // (versus stating it is UB) and *how* it would be enforced
                // is currently under discussion; see rust-lang/rust#58794.
                //
                // In either case, we mark item as explicitly nounwind.
                false
            }
        },
    );

    // Always annotate functions with the target-cpu they are compiled for.
    // Without this, ThinLTO won't inline Rust functions into Clang generated
    // functions (because Clang annotates functions this way too).
    apply_target_cpu_attr(cx, llfn);

    let features = llvm_target_features(cx.tcx.sess)
        .map(|s| s.to_string())
        .chain(codegen_fn_attrs.target_features.iter().map(|f| {
            let feature = &f.as_str();
            format!("+{}", llvm_util::to_llvm_feature(cx.tcx.sess, feature))
        }))
        .collect::<Vec<String>>()
        .join(",");

    if !features.is_empty() {
        let val = CString::new(features).unwrap();
        llvm::AddFunctionAttrStringValue(
            llfn,
            llvm::AttributePlace::Function,
            const_cstr!("target-features"),
            &val,
        );
    }

    // Note that currently the `wasm-import-module` doesn't do anything, but
    // eventually LLVM 7 should read this and ferry the appropriate import
    // module to the output file.
    if cx.tcx.sess.target.target.arch == "wasm32" {
        if let Some(module) = wasm_import_module(cx.tcx, instance.def_id()) {
            llvm::AddFunctionAttrStringValue(
                llfn,
                llvm::AttributePlace::Function,
                const_cstr!("wasm-import-module"),
                &module,
            );

            let name =
                codegen_fn_attrs.link_name.unwrap_or_else(|| cx.tcx.item_name(instance.def_id()));
            let name = CString::new(&name.as_str()[..]).unwrap();
            llvm::AddFunctionAttrStringValue(
                llfn,
                llvm::AttributePlace::Function,
                const_cstr!("wasm-import-name"),
                &name,
            );
        }
    }
}

pub fn provide(providers: &mut Providers<'_>) {
    providers.target_features_whitelist = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        if tcx.sess.opts.actually_rustdoc {
            // rustdoc needs to be able to document functions that use all the features, so
            // whitelist them all
            tcx.arena
                .alloc(llvm_util::all_known_features().map(|(a, b)| (a.to_string(), b)).collect())
        } else {
            tcx.arena.alloc(
                llvm_util::target_feature_whitelist(tcx.sess)
                    .iter()
                    .map(|&(a, b)| (a.to_string(), b))
                    .collect(),
            )
        }
    };

    provide_extern(providers);
}

pub fn provide_extern(providers: &mut Providers<'_>) {
    providers.wasm_import_module_map = |tcx, cnum| {
        // Build up a map from DefId to a `NativeLibrary` structure, where
        // `NativeLibrary` internally contains information about
        // `#[link(wasm_import_module = "...")]` for example.
        let native_libs = tcx.native_libraries(cnum);

        let def_id_to_native_lib = native_libs
            .iter()
            .filter_map(|lib| lib.foreign_module.map(|id| (id, lib)))
            .collect::<FxHashMap<_, _>>();

        let mut ret = FxHashMap::default();
        for lib in tcx.foreign_modules(cnum).iter() {
            let module = def_id_to_native_lib.get(&lib.def_id).and_then(|s| s.wasm_import_module);
            let module = match module {
                Some(s) => s,
                None => continue,
            };
            ret.extend(lib.foreign_items.iter().map(|id| {
                assert_eq!(id.krate, cnum);
                (*id, module.to_string())
            }));
        }

        tcx.arena.alloc(ret)
    };
}

fn wasm_import_module(tcx: TyCtxt<'_>, id: DefId) -> Option<CString> {
    tcx.wasm_import_module_map(id.krate).get(&id).map(|s| CString::new(&s[..]).unwrap())
}
