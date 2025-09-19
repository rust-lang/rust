//! Set and unset common attributes on LLVM values.
use rustc_hir::attrs::{InlineAttr, InstructionSetAttr, OptimizeAttr};
use rustc_hir::def_id::DefId;
use rustc_middle::middle::codegen_fn_attrs::{
    CodegenFnAttrFlags, CodegenFnAttrs, PatchableFunctionEntry,
};
use rustc_middle::ty::{self, TyCtxt};
use rustc_session::config::{BranchProtection, FunctionReturn, OptLevel, PAuthKey, PacRet};
use rustc_symbol_mangling::mangle_internal_symbol;
use rustc_target::spec::{FramePointer, SanitizerSet, StackProbeType, StackProtector};
use smallvec::SmallVec;

use crate::context::SimpleCx;
use crate::errors::SanitizerMemtagRequiresMte;
use crate::llvm::AttributePlace::Function;
use crate::llvm::{self, AllocKindFlags, Attribute, AttributeKind, AttributePlace, MemoryEffects};
use crate::value::Value;
use crate::{Session, attributes, llvm_util};

pub(crate) fn apply_to_llfn(llfn: &Value, idx: AttributePlace, attrs: &[&Attribute]) {
    if !attrs.is_empty() {
        llvm::AddFunctionAttributes(llfn, idx, attrs);
    }
}

pub(crate) fn apply_to_callsite(callsite: &Value, idx: AttributePlace, attrs: &[&Attribute]) {
    if !attrs.is_empty() {
        llvm::AddCallSiteAttributes(callsite, idx, attrs);
    }
}

/// Get LLVM attribute for the provided inline heuristic.
pub(crate) fn inline_attr<'ll, 'tcx>(
    cx: &SimpleCx<'ll>,
    tcx: TyCtxt<'tcx>,
    instance: ty::Instance<'tcx>,
) -> Option<&'ll Attribute> {
    // `optnone` requires `noinline`
    let codegen_fn_attrs = tcx.codegen_fn_attrs(instance.def_id());
    let inline = match (codegen_fn_attrs.inline, &codegen_fn_attrs.optimize) {
        (_, OptimizeAttr::DoNotOptimize) => InlineAttr::Never,
        (InlineAttr::None, _) if instance.def.requires_inline(tcx) => InlineAttr::Hint,
        (inline, _) => inline,
    };

    if !tcx.sess.opts.unstable_opts.inline_llvm {
        // disable LLVM inlining
        return Some(AttributeKind::NoInline.create_attr(cx.llcx));
    }
    match inline {
        InlineAttr::Hint => Some(AttributeKind::InlineHint.create_attr(cx.llcx)),
        InlineAttr::Always | InlineAttr::Force { .. } => {
            Some(AttributeKind::AlwaysInline.create_attr(cx.llcx))
        }
        InlineAttr::Never => {
            if tcx.sess.target.arch != "amdgpu" {
                Some(AttributeKind::NoInline.create_attr(cx.llcx))
            } else {
                None
            }
        }
        InlineAttr::None => None,
    }
}

#[inline]
fn patchable_function_entry_attrs<'ll>(
    cx: &SimpleCx<'ll>,
    sess: &Session,
    attr: Option<PatchableFunctionEntry>,
) -> SmallVec<[&'ll Attribute; 2]> {
    let mut attrs = SmallVec::new();
    let patchable_spec = attr.unwrap_or_else(|| {
        PatchableFunctionEntry::from_config(sess.opts.unstable_opts.patchable_function_entry)
    });
    let entry = patchable_spec.entry();
    let prefix = patchable_spec.prefix();
    if entry > 0 {
        attrs.push(llvm::CreateAttrStringValue(
            cx.llcx,
            "patchable-function-entry",
            &format!("{}", entry),
        ));
    }
    if prefix > 0 {
        attrs.push(llvm::CreateAttrStringValue(
            cx.llcx,
            "patchable-function-prefix",
            &format!("{}", prefix),
        ));
    }
    attrs
}

/// Get LLVM sanitize attributes.
#[inline]
pub(crate) fn sanitize_attrs<'ll, 'tcx>(
    cx: &SimpleCx<'ll>,
    tcx: TyCtxt<'tcx>,
    no_sanitize: SanitizerSet,
) -> SmallVec<[&'ll Attribute; 4]> {
    let mut attrs = SmallVec::new();
    let enabled = tcx.sess.opts.unstable_opts.sanitizer - no_sanitize;
    if enabled.contains(SanitizerSet::ADDRESS) || enabled.contains(SanitizerSet::KERNELADDRESS) {
        attrs.push(llvm::AttributeKind::SanitizeAddress.create_attr(cx.llcx));
    }
    if enabled.contains(SanitizerSet::MEMORY) {
        attrs.push(llvm::AttributeKind::SanitizeMemory.create_attr(cx.llcx));
    }
    if enabled.contains(SanitizerSet::THREAD) {
        attrs.push(llvm::AttributeKind::SanitizeThread.create_attr(cx.llcx));
    }
    if enabled.contains(SanitizerSet::HWADDRESS) {
        attrs.push(llvm::AttributeKind::SanitizeHWAddress.create_attr(cx.llcx));
    }
    if enabled.contains(SanitizerSet::SHADOWCALLSTACK) {
        attrs.push(llvm::AttributeKind::ShadowCallStack.create_attr(cx.llcx));
    }
    if enabled.contains(SanitizerSet::MEMTAG) {
        // Check to make sure the mte target feature is actually enabled.
        let features = tcx.global_backend_features(());
        let mte_feature =
            features.iter().map(|s| &s[..]).rfind(|n| ["+mte", "-mte"].contains(&&n[..]));
        if let None | Some("-mte") = mte_feature {
            tcx.dcx().emit_err(SanitizerMemtagRequiresMte);
        }

        attrs.push(llvm::AttributeKind::SanitizeMemTag.create_attr(cx.llcx));
    }
    if enabled.contains(SanitizerSet::SAFESTACK) {
        attrs.push(llvm::AttributeKind::SanitizeSafeStack.create_attr(cx.llcx));
    }
    attrs
}

/// Tell LLVM to emit or not emit the information necessary to unwind the stack for the function.
#[inline]
pub(crate) fn uwtable_attr(llcx: &llvm::Context, use_sync_unwind: Option<bool>) -> &Attribute {
    // NOTE: We should determine if we even need async unwind tables, as they
    // take have more overhead and if we can use sync unwind tables we
    // probably should.
    let async_unwind = !use_sync_unwind.unwrap_or(false);
    llvm::CreateUWTableAttr(llcx, async_unwind)
}

pub(crate) fn frame_pointer_type_attr<'ll>(
    cx: &SimpleCx<'ll>,
    sess: &Session,
) -> Option<&'ll Attribute> {
    let mut fp = sess.target.frame_pointer;
    let opts = &sess.opts;
    // "mcount" function relies on stack pointer.
    // See <https://sourceware.org/binutils/docs/gprof/Implementation.html>.
    if opts.unstable_opts.instrument_mcount {
        fp.ratchet(FramePointer::Always);
    }
    fp.ratchet(opts.cg.force_frame_pointers);
    let attr_value = match fp {
        FramePointer::Always => "all",
        FramePointer::NonLeaf => "non-leaf",
        FramePointer::MayOmit => return None,
    };
    Some(llvm::CreateAttrStringValue(cx.llcx, "frame-pointer", attr_value))
}

fn function_return_attr<'ll>(cx: &SimpleCx<'ll>, sess: &Session) -> Option<&'ll Attribute> {
    let function_return_attr = match sess.opts.unstable_opts.function_return {
        FunctionReturn::Keep => return None,
        FunctionReturn::ThunkExtern => AttributeKind::FnRetThunkExtern,
    };

    Some(function_return_attr.create_attr(cx.llcx))
}

/// Tell LLVM what instrument function to insert.
#[inline]
fn instrument_function_attr<'ll>(
    cx: &SimpleCx<'ll>,
    sess: &Session,
) -> SmallVec<[&'ll Attribute; 4]> {
    let mut attrs = SmallVec::new();
    if sess.opts.unstable_opts.instrument_mcount {
        // Similar to `clang -pg` behavior. Handled by the
        // `post-inline-ee-instrument` LLVM pass.

        // The function name varies on platforms.
        // See test/CodeGen/mcount.c in clang.
        let mcount_name = match &sess.target.llvm_mcount_intrinsic {
            Some(llvm_mcount_intrinsic) => llvm_mcount_intrinsic.as_ref(),
            None => sess.target.mcount.as_ref(),
        };

        attrs.push(llvm::CreateAttrStringValue(
            cx.llcx,
            "instrument-function-entry-inlined",
            mcount_name,
        ));
    }
    if let Some(options) = &sess.opts.unstable_opts.instrument_xray {
        // XRay instrumentation is similar to __cyg_profile_func_{enter,exit}.
        // Function prologue and epilogue are instrumented with NOP sleds,
        // a runtime library later replaces them with detours into tracing code.
        if options.always {
            attrs.push(llvm::CreateAttrStringValue(cx.llcx, "function-instrument", "xray-always"));
        }
        if options.never {
            attrs.push(llvm::CreateAttrStringValue(cx.llcx, "function-instrument", "xray-never"));
        }
        if options.ignore_loops {
            attrs.push(llvm::CreateAttrString(cx.llcx, "xray-ignore-loops"));
        }
        // LLVM will not choose the default for us, but rather requires specific
        // threshold in absence of "xray-always". Use the same default as Clang.
        let threshold = options.instruction_threshold.unwrap_or(200);
        attrs.push(llvm::CreateAttrStringValue(
            cx.llcx,
            "xray-instruction-threshold",
            &threshold.to_string(),
        ));
        if options.skip_entry {
            attrs.push(llvm::CreateAttrString(cx.llcx, "xray-skip-entry"));
        }
        if options.skip_exit {
            attrs.push(llvm::CreateAttrString(cx.llcx, "xray-skip-exit"));
        }
    }
    attrs
}

fn nojumptables_attr<'ll>(cx: &SimpleCx<'ll>, sess: &Session) -> Option<&'ll Attribute> {
    if !sess.opts.unstable_opts.no_jump_tables {
        return None;
    }

    Some(llvm::CreateAttrStringValue(cx.llcx, "no-jump-tables", "true"))
}

fn probestack_attr<'ll, 'tcx>(cx: &SimpleCx<'ll>, tcx: TyCtxt<'tcx>) -> Option<&'ll Attribute> {
    // Currently stack probes seem somewhat incompatible with the address
    // sanitizer and thread sanitizer. With asan we're already protected from
    // stack overflow anyway so we don't really need stack probes regardless.
    if tcx
        .sess
        .opts
        .unstable_opts
        .sanitizer
        .intersects(SanitizerSet::ADDRESS | SanitizerSet::THREAD)
    {
        return None;
    }

    // probestack doesn't play nice either with `-C profile-generate`.
    if tcx.sess.opts.cg.profile_generate.enabled() {
        return None;
    }

    let attr_value = match tcx.sess.target.stack_probes {
        StackProbeType::None => return None,
        // Request LLVM to generate the probes inline. If the given LLVM version does not support
        // this, no probe is generated at all (even if the attribute is specified).
        StackProbeType::Inline => "inline-asm",
        // Flag our internal `__rust_probestack` function as the stack probe symbol.
        // This is defined in the `compiler-builtins` crate for each architecture.
        StackProbeType::Call => &mangle_internal_symbol(tcx, "__rust_probestack"),
        // Pick from the two above based on the LLVM version.
        StackProbeType::InlineOrCall { min_llvm_version_for_inline } => {
            if llvm_util::get_version() < min_llvm_version_for_inline {
                &mangle_internal_symbol(tcx, "__rust_probestack")
            } else {
                "inline-asm"
            }
        }
    };
    Some(llvm::CreateAttrStringValue(cx.llcx, "probe-stack", attr_value))
}

fn stackprotector_attr<'ll>(cx: &SimpleCx<'ll>, sess: &Session) -> Option<&'ll Attribute> {
    let sspattr = match sess.stack_protector() {
        StackProtector::None => return None,
        StackProtector::All => AttributeKind::StackProtectReq,
        StackProtector::Strong => AttributeKind::StackProtectStrong,
        StackProtector::Basic => AttributeKind::StackProtect,
    };

    Some(sspattr.create_attr(cx.llcx))
}

fn backchain_attr<'ll>(cx: &SimpleCx<'ll>, sess: &Session) -> Option<&'ll Attribute> {
    if sess.target.arch != "s390x" {
        return None;
    }

    let requested_features = sess.opts.cg.target_feature.split(',');
    let found_positive = requested_features.clone().any(|r| r == "+backchain");

    if found_positive { Some(llvm::CreateAttrString(cx.llcx, "backchain")) } else { None }
}

pub(crate) fn target_cpu_attr<'ll>(cx: &SimpleCx<'ll>, sess: &Session) -> &'ll Attribute {
    let target_cpu = llvm_util::target_cpu(sess);
    llvm::CreateAttrStringValue(cx.llcx, "target-cpu", target_cpu)
}

pub(crate) fn tune_cpu_attr<'ll>(cx: &SimpleCx<'ll>, sess: &Session) -> Option<&'ll Attribute> {
    llvm_util::tune_cpu(sess)
        .map(|tune_cpu| llvm::CreateAttrStringValue(cx.llcx, "tune-cpu", tune_cpu))
}

/// Get the `target-features` LLVM attribute.
pub(crate) fn target_features_attr<'ll, 'tcx>(
    cx: &SimpleCx<'ll>,
    tcx: TyCtxt<'tcx>,
    function_features: Vec<String>,
) -> Option<&'ll Attribute> {
    let global_features = tcx.global_backend_features(()).iter().map(String::as_str);
    let function_features = function_features.iter().map(String::as_str);
    let target_features =
        global_features.chain(function_features).intersperse(",").collect::<String>();
    (!target_features.is_empty())
        .then(|| llvm::CreateAttrStringValue(cx.llcx, "target-features", &target_features))
}

/// Get the `NonLazyBind` LLVM attribute,
/// if the codegen options allow skipping the PLT.
pub(crate) fn non_lazy_bind_attr<'ll>(
    cx: &SimpleCx<'ll>,
    sess: &Session,
) -> Option<&'ll Attribute> {
    // Don't generate calls through PLT if it's not necessary
    if !sess.needs_plt() { Some(AttributeKind::NonLazyBind.create_attr(cx.llcx)) } else { None }
}

/// Get the default optimizations attrs for a function.
#[inline]
pub(crate) fn default_optimisation_attrs<'ll>(
    cx: &SimpleCx<'ll>,
    sess: &Session,
) -> SmallVec<[&'ll Attribute; 2]> {
    let mut attrs = SmallVec::new();
    match sess.opts.optimize {
        OptLevel::Size => {
            attrs.push(llvm::AttributeKind::OptimizeForSize.create_attr(cx.llcx));
        }
        OptLevel::SizeMin => {
            attrs.push(llvm::AttributeKind::MinSize.create_attr(cx.llcx));
            attrs.push(llvm::AttributeKind::OptimizeForSize.create_attr(cx.llcx));
        }
        _ => {}
    }
    attrs
}

fn create_alloc_family_attr(llcx: &llvm::Context) -> &llvm::Attribute {
    llvm::CreateAttrStringValue(llcx, "alloc-family", "__rust_alloc")
}

/// Helper for `FnAbi::apply_attrs_llfn`:
/// Composite function which sets LLVM attributes for function depending on its AST (`#[attribute]`)
/// attributes.
pub(crate) fn llfn_attrs_from_instance<'ll, 'tcx>(
    cx: &SimpleCx<'ll>,
    tcx: TyCtxt<'tcx>,
    llfn: &'ll Value,
    codegen_fn_attrs: &CodegenFnAttrs,
    instance: Option<ty::Instance<'tcx>>,
) {
    let sess = tcx.sess;
    let mut to_add = SmallVec::<[_; 16]>::new();

    match codegen_fn_attrs.optimize {
        OptimizeAttr::Default => {
            to_add.extend(default_optimisation_attrs(cx, sess));
        }
        OptimizeAttr::DoNotOptimize => {
            to_add.push(llvm::AttributeKind::OptimizeNone.create_attr(cx.llcx));
        }
        OptimizeAttr::Size => {
            to_add.push(llvm::AttributeKind::MinSize.create_attr(cx.llcx));
            to_add.push(llvm::AttributeKind::OptimizeForSize.create_attr(cx.llcx));
        }
        OptimizeAttr::Speed => {}
    }

    if sess.must_emit_unwind_tables() {
        to_add.push(uwtable_attr(cx.llcx, sess.opts.unstable_opts.use_sync_unwind));
    }

    if sess.opts.unstable_opts.profile_sample_use.is_some() {
        to_add.push(llvm::CreateAttrString(cx.llcx, "use-sample-profile"));
    }

    // FIXME: none of these functions interact with source level attributes.
    to_add.extend(frame_pointer_type_attr(cx, sess));
    to_add.extend(function_return_attr(cx, sess));
    to_add.extend(instrument_function_attr(cx, sess));
    to_add.extend(nojumptables_attr(cx, sess));
    to_add.extend(probestack_attr(cx, tcx));
    to_add.extend(stackprotector_attr(cx, sess));

    if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::NO_BUILTINS) {
        to_add.push(llvm::CreateAttrString(cx.llcx, "no-builtins"));
    }

    if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::COLD) {
        to_add.push(AttributeKind::Cold.create_attr(cx.llcx));
    }
    if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::FFI_PURE) {
        to_add.push(MemoryEffects::ReadOnly.create_attr(cx.llcx));
    }
    if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::FFI_CONST) {
        to_add.push(MemoryEffects::None.create_attr(cx.llcx));
    }
    if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::NAKED) {
        // do nothing; a naked function is converted into an extern function
        // and a global assembly block. LLVM's support for naked functions is
        // not used.
    } else {
        // Do not set sanitizer attributes for naked functions.
        to_add.extend(sanitize_attrs(cx, tcx, codegen_fn_attrs.no_sanitize));

        // For non-naked functions, set branch protection attributes on aarch64.
        if let Some(BranchProtection { bti, pac_ret, gcs }) =
            sess.opts.unstable_opts.branch_protection
        {
            assert!(sess.target.arch == "aarch64");
            if bti {
                to_add.push(llvm::CreateAttrString(cx.llcx, "branch-target-enforcement"));
            }
            if gcs {
                to_add.push(llvm::CreateAttrString(cx.llcx, "guarded-control-stack"));
            }
            if let Some(PacRet { leaf, pc, key }) = pac_ret {
                if pc {
                    to_add.push(llvm::CreateAttrString(cx.llcx, "branch-protection-pauth-lr"));
                }
                to_add.push(llvm::CreateAttrStringValue(
                    cx.llcx,
                    "sign-return-address",
                    if leaf { "all" } else { "non-leaf" },
                ));
                to_add.push(llvm::CreateAttrStringValue(
                    cx.llcx,
                    "sign-return-address-key",
                    if key == PAuthKey::A { "a_key" } else { "b_key" },
                ));
            }
        }
    }
    if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::ALLOCATOR)
        || codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::ALLOCATOR_ZEROED)
    {
        to_add.push(create_alloc_family_attr(cx.llcx));
        if let Some(instance) = instance
            && let Some(zv) =
                tcx.get_attr(instance.def_id(), rustc_span::sym::rustc_allocator_zeroed_variant)
            && let Some(name) = zv.value_str()
        {
            to_add.push(llvm::CreateAttrStringValue(
                cx.llcx,
                "alloc-variant-zeroed",
                &mangle_internal_symbol(tcx, name.as_str()),
            ));
        }
        // apply to argument place instead of function
        let alloc_align = AttributeKind::AllocAlign.create_attr(cx.llcx);
        attributes::apply_to_llfn(llfn, AttributePlace::Argument(1), &[alloc_align]);
        to_add.push(llvm::CreateAllocSizeAttr(cx.llcx, 0));
        let mut flags = AllocKindFlags::Alloc | AllocKindFlags::Aligned;
        if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::ALLOCATOR) {
            flags |= AllocKindFlags::Uninitialized;
        } else {
            flags |= AllocKindFlags::Zeroed;
        }
        to_add.push(llvm::CreateAllocKindAttr(cx.llcx, flags));
        // apply to return place instead of function (unlike all other attributes applied in this
        // function)
        let no_alias = AttributeKind::NoAlias.create_attr(cx.llcx);
        attributes::apply_to_llfn(llfn, AttributePlace::ReturnValue, &[no_alias]);
    }
    if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::REALLOCATOR) {
        to_add.push(create_alloc_family_attr(cx.llcx));
        to_add.push(llvm::CreateAllocKindAttr(
            cx.llcx,
            AllocKindFlags::Realloc | AllocKindFlags::Aligned,
        ));
        // applies to argument place instead of function place
        let allocated_pointer = AttributeKind::AllocatedPointer.create_attr(cx.llcx);
        attributes::apply_to_llfn(llfn, AttributePlace::Argument(0), &[allocated_pointer]);
        // apply to argument place instead of function
        let alloc_align = AttributeKind::AllocAlign.create_attr(cx.llcx);
        attributes::apply_to_llfn(llfn, AttributePlace::Argument(2), &[alloc_align]);
        to_add.push(llvm::CreateAllocSizeAttr(cx.llcx, 3));
        let no_alias = AttributeKind::NoAlias.create_attr(cx.llcx);
        attributes::apply_to_llfn(llfn, AttributePlace::ReturnValue, &[no_alias]);
    }
    if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::DEALLOCATOR) {
        to_add.push(create_alloc_family_attr(cx.llcx));
        to_add.push(llvm::CreateAllocKindAttr(cx.llcx, AllocKindFlags::Free));
        // applies to argument place instead of function place
        let allocated_pointer = AttributeKind::AllocatedPointer.create_attr(cx.llcx);
        attributes::apply_to_llfn(llfn, AttributePlace::Argument(0), &[allocated_pointer]);
    }
    if let Some(align) = codegen_fn_attrs.alignment {
        llvm::set_alignment(llfn, align);
    }
    if let Some(backchain) = backchain_attr(cx, sess) {
        to_add.push(backchain);
    }
    to_add.extend(patchable_function_entry_attrs(
        cx,
        sess,
        codegen_fn_attrs.patchable_function_entry,
    ));

    // Always annotate functions with the target-cpu they are compiled for.
    // Without this, ThinLTO won't inline Rust functions into Clang generated
    // functions (because Clang annotates functions this way too).
    to_add.push(target_cpu_attr(cx, sess));
    // tune-cpu is only conveyed through the attribute for our purpose.
    // The target doesn't care; the subtarget reads our attribute.
    to_add.extend(tune_cpu_attr(cx, sess));

    let function_features =
        codegen_fn_attrs.target_features.iter().map(|f| f.name.as_str()).collect::<Vec<&str>>();

    // Apply function attributes as per usual if there are no user defined
    // target features otherwise this will get applied at the callsite.
    if function_features.is_empty() {
        if let Some(instance) = instance
            && let Some(inline_attr) = inline_attr(cx, tcx, instance)
        {
            to_add.push(inline_attr);
        }
    }

    let function_features = function_features
        .iter()
        // Convert to LLVMFeatures and filter out unavailable ones
        .flat_map(|feat| llvm_util::to_llvm_features(sess, feat))
        // Convert LLVMFeatures & dependencies to +<feats>s
        .flat_map(|feat| feat.into_iter().map(|f| format!("+{f}")))
        .chain(codegen_fn_attrs.instruction_set.iter().map(|x| match x {
            InstructionSetAttr::ArmA32 => "-thumb-mode".to_string(),
            InstructionSetAttr::ArmT32 => "+thumb-mode".to_string(),
        }))
        .collect::<Vec<String>>();

    if sess.target.is_like_wasm {
        // If this function is an import from the environment but the wasm
        // import has a specific module/name, apply them here.
        if let Some(instance) = instance
            && let Some(module) = wasm_import_module(tcx, instance.def_id())
        {
            to_add.push(llvm::CreateAttrStringValue(cx.llcx, "wasm-import-module", module));

            let name =
                codegen_fn_attrs.symbol_name.unwrap_or_else(|| tcx.item_name(instance.def_id()));
            let name = name.as_str();
            to_add.push(llvm::CreateAttrStringValue(cx.llcx, "wasm-import-name", name));
        }
    }

    to_add.extend(target_features_attr(cx, tcx, function_features));

    attributes::apply_to_llfn(llfn, Function, &to_add);
}

fn wasm_import_module(tcx: TyCtxt<'_>, id: DefId) -> Option<&String> {
    tcx.wasm_import_module_map(id.krate).get(&id)
}
