//! Set and unset common attributes on LLVM values.

use std::ffi::CString;

use rustc::hir::{CodegenFnAttrFlags, CodegenFnAttrs};
use rustc::hir::def_id::{DefId, LOCAL_CRATE};
use rustc::session::Session;
use rustc::session::config::{Sanitizer, OptLevel};
use rustc::ty::{self, TyCtxt, PolyFnSig};
use rustc::ty::layout::HasTyCtxt;
use rustc::ty::query::Providers;
use rustc_data_structures::small_c_str::SmallCStr;
use rustc_data_structures::fx::FxHashMap;
use rustc_target::spec::PanicStrategy;
use rustc_codegen_ssa::traits::*;

use crate::abi::Abi;
use crate::attributes;
use crate::llvm::{self, Attribute};
use crate::llvm::AttributePlace::Function;
use crate::llvm_util;
pub use syntax::attr::{self, InlineAttr, OptimizeAttr};

use crate::context::CodegenCx;
use crate::value::Value;

/// Mark LLVM function to use provided inline heuristic.
#[inline]
pub fn inline(cx: &CodegenCx<'ll, '_>, val: &'ll Value, inline: InlineAttr) {
    use self::InlineAttr::*;
    match inline {
        Hint   => Attribute::InlineHint.apply_llfn(Function, val),
        Always => Attribute::AlwaysInline.apply_llfn(Function, val),
        Never  => {
            if cx.tcx().sess.target.target.arch != "amdgpu" {
                Attribute::NoInline.apply_llfn(Function, val);
            }
        },
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
fn unwind(val: &'ll Value, can_unwind: bool) {
    Attribute::NoUnwind.toggle_llfn(Function, val, !can_unwind);
}

/// Tell LLVM if this function should be 'naked', i.e., skip the epilogue and prologue.
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

/// Tell LLVM what instrument function to insert.
#[inline]
pub fn set_instrument_function(cx: &CodegenCx<'ll, '_>, llfn: &'ll Value) {
    if cx.sess().instrument_mcount() {
        // Similar to `clang -pg` behavior. Handled by the
        // `post-inline-ee-instrument` LLVM pass.

        // The function name varies on platforms.
        // See test/CodeGen/mcount.c in clang.
        let mcount_name = CString::new(
            cx.sess().target.target.options.target_mcount.as_str().as_bytes()).unwrap();

        llvm::AddFunctionAttrStringValue(
            llfn, llvm::AttributePlace::Function,
            const_cstr!("instrument-function-entry-inlined"), &mcount_name);
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
    if let Some(Sanitizer::Address) = cx.sess().opts.debugging_opts.sanitizer {
        return
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
    let target_cpu = SmallCStr::new(llvm_util::target_cpu(cx.tcx.sess));
    llvm::AddFunctionAttrStringValue(
            llfn,
            llvm::AttributePlace::Function,
            const_cstr!("target-cpu"),
            target_cpu.as_c_str());
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
        },
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
    id: Option<DefId>,
    sig: PolyFnSig<'tcx>,
) {
    let codegen_fn_attrs = id.map(|id| cx.tcx.codegen_fn_attrs(id))
        .unwrap_or_else(|| CodegenFnAttrs::new());

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
    if !cx.sess().no_landing_pads() ||
       cx.sess().target.target.options.requires_uwtable {
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
        Attribute::NoAlias.apply_llfn(
            llvm::AttributePlace::ReturnValue, llfn);
    }

    unwind(llfn, if cx.tcx.sess.panic_strategy() != PanicStrategy::Unwind {
        // In panic=abort mode we assume nothing can unwind anywhere, so
        // optimize based on this!
        false
    } else if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::UNWIND) {
        // If a specific #[unwind] attribute is present, use that
        true
    } else if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::RUSTC_ALLOCATOR_NOUNWIND) {
        // Special attribute for allocator functions, which can't unwind
        false
    } else if let Some(id) = id {
        let sig = cx.tcx.normalize_erasing_late_bound_regions(ty::ParamEnv::reveal_all(), &sig);
        if cx.tcx.is_foreign_item(id) {
            // Foreign items like `extern "C" { fn foo(); }` are assumed not to
            // unwind
            false
        } else if sig.abi != Abi::Rust && sig.abi != Abi::RustCall {
            // Any items defined in Rust that *don't* have the `extern` ABI are
            // defined to not unwind. We insert shims to abort if an unwind
            // happens to enforce this.
            false
        } else {
            // Anything else defined in Rust is assumed that it can possibly
            // unwind
            true
        }
    } else {
        // assume this can possibly unwind, avoiding the application of a
        // `nounwind` attribute below.
        true
    });

    // Always annotate functions with the target-cpu they are compiled for.
    // Without this, ThinLTO won't inline Rust functions into Clang generated
    // functions (because Clang annotates functions this way too).
    apply_target_cpu_attr(cx, llfn);

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

pub fn provide(providers: &mut Providers<'_>) {
    providers.target_features_whitelist = |tcx, cnum| {
        assert_eq!(cnum, LOCAL_CRATE);
        if tcx.sess.opts.actually_rustdoc {
            // rustdoc needs to be able to document functions that use all the features, so
            // whitelist them all
            tcx.arena.alloc(llvm_util::all_known_features()
                .map(|(a, b)| (a.to_string(), b))
                .collect())
        } else {
            tcx.arena.alloc(llvm_util::target_feature_whitelist(tcx.sess)
                .iter()
                .map(|&(a, b)| (a.to_string(), b))
                .collect())
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

        let def_id_to_native_lib = native_libs.iter().filter_map(|lib|
            if let Some(id) = lib.foreign_module {
                Some((id, lib))
            } else {
                None
            }
        ).collect::<FxHashMap<_, _>>();

        let mut ret = FxHashMap::default();
        for lib in tcx.foreign_modules(cnum).iter() {
            let module = def_id_to_native_lib
                .get(&lib.def_id)
                .and_then(|s| s.wasm_import_module);
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
    tcx.wasm_import_module_map(id.krate)
        .get(&id)
        .map(|s| CString::new(&s[..]).unwrap())
}
