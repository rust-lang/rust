//! Set and unset common attributes on LLVM values.

use std::ffi::CString;

use cstr::cstr;
use rustc_codegen_ssa::traits::*;
use rustc_data_structures::small_c_str::SmallCStr;
use rustc_hir::def_id::DefId;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::ty::layout::HasTyCtxt;
use rustc_middle::ty::{self, TyCtxt};
use rustc_session::config::OptLevel;
use rustc_session::Session;
use rustc_target::spec::abi::Abi;
use rustc_target::spec::{FramePointer, SanitizerSet, StackProbeType, StackProtector};

use crate::attributes;
use crate::llvm::AttributePlace::Function;
use crate::llvm::{self, Attribute};
use crate::llvm_util;
pub use rustc_attr::{InlineAttr, InstructionSetAttr, OptimizeAttr};

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
            if cx.tcx().sess.target.arch != "amdgpu" {
                Attribute::NoInline.apply_llfn(Function, val);
            }
        }
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
    if enabled.contains(SanitizerSet::HWADDRESS) {
        llvm::Attribute::SanitizeHWAddress.apply_llfn(Function, llfn);
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

pub fn set_frame_pointer_type(cx: &CodegenCx<'ll, '_>, llfn: &'ll Value) {
    let mut fp = cx.sess().target.frame_pointer;
    // "mcount" function relies on stack pointer.
    // See <https://sourceware.org/binutils/docs/gprof/Implementation.html>.
    if cx.sess().instrument_mcount() || matches!(cx.sess().opts.cg.force_frame_pointers, Some(true))
    {
        fp = FramePointer::Always;
    }
    let attr_value = match fp {
        FramePointer::Always => cstr!("all"),
        FramePointer::NonLeaf => cstr!("non-leaf"),
        FramePointer::MayOmit => return,
    };
    llvm::AddFunctionAttrStringValue(
        llfn,
        llvm::AttributePlace::Function,
        cstr!("frame-pointer"),
        attr_value,
    );
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
            cstr!("instrument-function-entry-inlined"),
            &mcount_name,
        );
    }
}

fn set_probestack(cx: &CodegenCx<'ll, '_>, llfn: &'ll Value) {
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

    let attr_value = match cx.sess().target.stack_probes {
        StackProbeType::None => None,
        // Request LLVM to generate the probes inline. If the given LLVM version does not support
        // this, no probe is generated at all (even if the attribute is specified).
        StackProbeType::Inline => Some(cstr!("inline-asm")),
        // Flag our internal `__rust_probestack` function as the stack probe symbol.
        // This is defined in the `compiler-builtins` crate for each architecture.
        StackProbeType::Call => Some(cstr!("__rust_probestack")),
        // Pick from the two above based on the LLVM version.
        StackProbeType::InlineOrCall { min_llvm_version_for_inline } => {
            if llvm_util::get_version() < min_llvm_version_for_inline {
                Some(cstr!("__rust_probestack"))
            } else {
                Some(cstr!("inline-asm"))
            }
        }
    };
    if let Some(attr_value) = attr_value {
        llvm::AddFunctionAttrStringValue(
            llfn,
            llvm::AttributePlace::Function,
            cstr!("probe-stack"),
            attr_value,
        );
    }
}

fn set_stackprotector(cx: &CodegenCx<'ll, '_>, llfn: &'ll Value) {
    let sspattr = match cx.sess().stack_protector() {
        StackProtector::None => return,
        StackProtector::All => Attribute::StackProtectReq,
        StackProtector::Strong => Attribute::StackProtectStrong,
        StackProtector::Basic => Attribute::StackProtect,
    };

    sspattr.apply_llfn(Function, llfn)
}

pub fn apply_target_cpu_attr(cx: &CodegenCx<'ll, '_>, llfn: &'ll Value) {
    let target_cpu = SmallCStr::new(llvm_util::target_cpu(cx.tcx.sess));
    llvm::AddFunctionAttrStringValue(
        llfn,
        llvm::AttributePlace::Function,
        cstr!("target-cpu"),
        target_cpu.as_c_str(),
    );
}

pub fn apply_tune_cpu_attr(cx: &CodegenCx<'ll, '_>, llfn: &'ll Value) {
    if let Some(tune) = llvm_util::tune_cpu(cx.tcx.sess) {
        let tune_cpu = SmallCStr::new(tune);
        llvm::AddFunctionAttrStringValue(
            llfn,
            llvm::AttributePlace::Function,
            cstr!("tune-cpu"),
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

    let inline_attr = if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::NAKED) {
        InlineAttr::Never
    } else if codegen_fn_attrs.inline == InlineAttr::None && instance.def.requires_inline(cx.tcx) {
        InlineAttr::Hint
    } else {
        codegen_fn_attrs.inline
    };
    inline(cx, llfn, inline_attr);

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

    if cx.sess().opts.debugging_opts.profile_sample_use.is_some() {
        llvm::AddFunctionAttrString(llfn, Function, cstr!("use-sample-profile"));
    }

    // FIXME: none of these three functions interact with source level attributes.
    set_frame_pointer_type(cx, llfn);
    set_instrument_function(cx, llfn);
    set_probestack(cx, llfn);
    set_stackprotector(cx, llfn);

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
        llvm::AddFunctionAttrString(llfn, Function, cstr!("cmse_nonsecure_entry"));
    }
    if let Some(align) = codegen_fn_attrs.alignment {
        llvm::set_alignment(llfn, align as usize);
    }
    sanitize(cx, codegen_fn_attrs.no_sanitize, llfn);

    // Always annotate functions with the target-cpu they are compiled for.
    // Without this, ThinLTO won't inline Rust functions into Clang generated
    // functions (because Clang annotates functions this way too).
    apply_target_cpu_attr(cx, llfn);
    // tune-cpu is only conveyed through the attribute for our purpose.
    // The target doesn't care; the subtarget reads our attribute.
    apply_tune_cpu_attr(cx, llfn);

    let mut function_features = codegen_fn_attrs
        .target_features
        .iter()
        .flat_map(|f| {
            let feature = &f.as_str();
            llvm_util::to_llvm_feature(cx.tcx.sess, feature)
                .into_iter()
                .map(|f| format!("+{}", f))
                .collect::<Vec<String>>()
        })
        .chain(codegen_fn_attrs.instruction_set.iter().map(|x| match x {
            InstructionSetAttr::ArmA32 => "-thumb-mode".to_string(),
            InstructionSetAttr::ArmT32 => "+thumb-mode".to_string(),
        }))
        .collect::<Vec<String>>();

    if cx.tcx.sess.target.is_like_wasm {
        // If this function is an import from the environment but the wasm
        // import has a specific module/name, apply them here.
        if let Some(module) = wasm_import_module(cx.tcx, instance.def_id()) {
            llvm::AddFunctionAttrStringValue(
                llfn,
                llvm::AttributePlace::Function,
                cstr!("wasm-import-module"),
                &module,
            );

            let name =
                codegen_fn_attrs.link_name.unwrap_or_else(|| cx.tcx.item_name(instance.def_id()));
            let name = CString::new(&name.as_str()[..]).unwrap();
            llvm::AddFunctionAttrStringValue(
                llfn,
                llvm::AttributePlace::Function,
                cstr!("wasm-import-name"),
                &name,
            );
        }

        // The `"wasm"` abi on wasm targets automatically enables the
        // `+multivalue` feature because the purpose of the wasm abi is to match
        // the WebAssembly specification, which has this feature. This won't be
        // needed when LLVM enables this `multivalue` feature by default.
        if !cx.tcx.is_closure(instance.def_id()) {
            let abi = cx.tcx.fn_sig(instance.def_id()).abi();
            if abi == Abi::Wasm {
                function_features.push("+multivalue".to_string());
            }
        }
    }

    if !function_features.is_empty() {
        let mut global_features = llvm_util::llvm_global_features(cx.tcx.sess);
        global_features.extend(function_features.into_iter());
        let features = global_features.join(",");
        let val = CString::new(features).unwrap();
        llvm::AddFunctionAttrStringValue(
            llfn,
            llvm::AttributePlace::Function,
            cstr!("target-features"),
            &val,
        );
    }
}

fn wasm_import_module(tcx: TyCtxt<'_>, id: DefId) -> Option<CString> {
    tcx.wasm_import_module_map(id.krate).get(&id).map(|s| CString::new(&s[..]).unwrap())
}
