//! Set and unset common attributes on LLVM values.

use std::ffi::CString;

use rustc_codegen_ssa::traits::*;
use rustc_data_structures::const_cstr;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::small_c_str::SmallCStr;
use rustc_hir::def_id::DefId;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::ty::layout::HasTyCtxt;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::{self, TyCtxt};
use rustc_session::config::{OptLevel, SanitizerSet};
use rustc_session::Session;

use crate::attributes;
use crate::llvm::AttributePlace::Function;
use crate::llvm::{self, Attribute};
use crate::llvm_util;
pub use rustc_attr::{InlineAttr, InstructionSetAttr, OptimizeAttr};

use crate::context::CodegenCx;
use crate::value::Value;

/// Mark LLVM function to use provided inline heuristic.
#[inline]
fn inline(cx: &CodegenCx<'ll, '_>, val: &'ll Value, inline: InlineAttr, requires_inline: bool) {
    use self::InlineAttr::*;
    match inline {
        Hint => Attribute::InlineHint.apply_llfn(Function, val),
        Always => Attribute::AlwaysInline.apply_llfn(Function, val),
        Never => {
            if cx.tcx().sess.target.arch != "amdgpu" {
                Attribute::NoInline.apply_llfn(Function, val);
            }
        }
        None if requires_inline => Attribute::InlineHint.apply_llfn(Function, val),
        None => {}
    };
}

/// Apply LLVM sanitize attributes.
#[inline]
pub fn sanitize(cx: &CodegenCx<'ll, '_>, no_sanitize: SanitizerSet, llfn: &'ll Value) {
    let enabled = cx.tcx.sess.opts.debugging_opts.sanitizer - no_sanitize;
    if enabled.contains(SanitizerSet::ADDRESS) {
        llvm::Attribute::SanitizeAddress.apply_llfn(Function, llfn);
    }
    if enabled.contains(SanitizerSet::MEMORY) {
        llvm::Attribute::SanitizeMemory.apply_llfn(Function, llfn);
    }
    if enabled.contains(SanitizerSet::THREAD) {
        llvm::Attribute::SanitizeThread.apply_llfn(Function, llfn);
    }
}

/// Tell LLVM to emit or not emit the information necessary to unwind the stack for the function.
#[inline]
pub fn emit_uwtable(val: &'ll Value, emit: bool) {
    Attribute::UWTable.toggle_llfn(Function, val, emit);
}

/// Tell LLVM if this function should be 'naked', i.e., skip the epilogue and prologue.
#[inline]
fn naked(val: &'ll Value, is_naked: bool) {
    Attribute::Naked.toggle_llfn(Function, val, is_naked);
}

pub fn set_frame_pointer_elimination(cx: &CodegenCx<'ll, '_>, llfn: &'ll Value) {
    if cx.sess().must_not_eliminate_frame_pointers() {
        llvm::AddFunctionAttrStringValue(
            llfn,
            llvm::AttributePlace::Function,
            const_cstr!("frame-pointer"),
            const_cstr!("all"),
        );
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
        let mcount_name = CString::new(cx.sess().target.mcount.as_str().as_bytes()).unwrap();

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
    if !cx.sess().target.stack_probes {
        return;
    }

    // Currently stack probes seem somewhat incompatible with the address
    // sanitizer and thread sanitizer. With asan we're already protected from
    // stack overflow anyway so we don't really need stack probes regardless.
    if cx
        .sess()
        .opts
        .debugging_opts
        .sanitizer
        .intersects(SanitizerSet::ADDRESS | SanitizerSet::THREAD)
    {
        return;
    }

    // probestack doesn't play nice either with `-C profile-generate`.
    if cx.sess().opts.cg.profile_generate.enabled() {
        return;
    }

    // probestack doesn't play nice either with gcov profiling.
    if cx.sess().opts.debugging_opts.profile {
        return;
    }

    // FIXME(richkadel): Make sure probestack plays nice with `-Z instrument-coverage`
    // or disable it if not, similar to above early exits.

    // Flag our internal `__rust_probestack` function as the stack probe symbol.
    // This is defined in the `compiler-builtins` crate for each architecture.
    llvm::AddFunctionAttrStringValue(
        llfn,
        llvm::AttributePlace::Function,
        const_cstr!("probe-stack"),
        const_cstr!("__rust_probestack"),
    );
}

pub fn llvm_target_features(sess: &Session) -> impl Iterator<Item = &str> {
    const RUSTC_SPECIFIC_FEATURES: &[&str] = &["crt-static"];

    let cmdline = sess
        .opts
        .cg
        .target_feature
        .split(',')
        .filter(|f| !RUSTC_SPECIFIC_FEATURES.iter().any(|s| f.contains(s)));
    sess.target.features.split(',').chain(cmdline).filter(|l| !l.is_empty())
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

pub fn apply_tune_cpu_attr(cx: &CodegenCx<'ll, '_>, llfn: &'ll Value) {
    if let Some(tune) = llvm_util::tune_cpu(cx.tcx.sess) {
        let tune_cpu = SmallCStr::new(tune);
        llvm::AddFunctionAttrStringValue(
            llfn,
            llvm::AttributePlace::Function,
            const_cstr!("tune-cpu"),
            tune_cpu.as_c_str(),
        );
    }
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
pub fn from_fn_attrs(cx: &CodegenCx<'ll, 'tcx>, llfn: &'ll Value, instance: ty::Instance<'tcx>) {
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

    inline(cx, llfn, codegen_fn_attrs.inline.clone(), instance.def.requires_inline(cx.tcx));

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
    // You can also find more info on why Windows always requires uwtables here:
    //      https://bugzilla.mozilla.org/show_bug.cgi?id=1302078
    if cx.sess().must_emit_unwind_tables() {
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
    if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::FFI_PURE) {
        Attribute::ReadOnly.apply_llfn(Function, llfn);
    }
    if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::FFI_CONST) {
        Attribute::ReadNone.apply_llfn(Function, llfn);
    }
    if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::NAKED) {
        naked(llfn, true);
    }
    if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::ALLOCATOR) {
        Attribute::NoAlias.apply_llfn(llvm::AttributePlace::ReturnValue, llfn);
    }
    if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::CMSE_NONSECURE_ENTRY) {
        llvm::AddFunctionAttrString(llfn, Function, const_cstr!("cmse_nonsecure_entry"));
    }
    sanitize(cx, codegen_fn_attrs.no_sanitize, llfn);

    // Always annotate functions with the target-cpu they are compiled for.
    // Without this, ThinLTO won't inline Rust functions into Clang generated
    // functions (because Clang annotates functions this way too).
    apply_target_cpu_attr(cx, llfn);
    // tune-cpu is only conveyed through the attribute for our purpose.
    // The target doesn't care; the subtarget reads our attribute.
    apply_tune_cpu_attr(cx, llfn);

    let features = llvm_target_features(cx.tcx.sess)
        .map(|s| s.to_string())
        .chain(codegen_fn_attrs.target_features.iter().map(|f| {
            let feature = &f.as_str();
            format!("+{}", llvm_util::to_llvm_feature(cx.tcx.sess, feature))
        }))
        .chain(codegen_fn_attrs.instruction_set.iter().map(|x| match x {
            InstructionSetAttr::ArmA32 => "-thumb-mode".to_string(),
            InstructionSetAttr::ArmT32 => "+thumb-mode".to_string(),
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
    if cx.tcx.sess.target.arch == "wasm32" {
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

pub fn provide_both(providers: &mut Providers) {
    providers.wasm_import_module_map = |tcx, cnum| {
        // Build up a map from DefId to a `NativeLib` structure, where
        // `NativeLib` internally contains information about
        // `#[link(wasm_import_module = "...")]` for example.
        let native_libs = tcx.native_libraries(cnum);

        let def_id_to_native_lib = native_libs
            .iter()
            .filter_map(|lib| lib.foreign_module.map(|id| (id, lib)))
            .collect::<FxHashMap<_, _>>();

        let mut ret = FxHashMap::default();
        for (def_id, lib) in tcx.foreign_modules(cnum).iter() {
            let module = def_id_to_native_lib.get(&def_id).and_then(|s| s.wasm_import_module);
            let module = match module {
                Some(s) => s,
                None => continue,
            };
            ret.extend(lib.foreign_items.iter().map(|id| {
                assert_eq!(id.krate, cnum);
                (*id, module.to_string())
            }));
        }

        ret
    };
}

fn wasm_import_module(tcx: TyCtxt<'_>, id: DefId) -> Option<CString> {
    tcx.wasm_import_module_map(id.krate).get(&id).map(|s| CString::new(&s[..]).unwrap())
}
