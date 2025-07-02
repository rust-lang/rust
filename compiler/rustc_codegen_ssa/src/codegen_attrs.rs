use std::str::FromStr;

use rustc_abi::{Align, ExternAbi};
use rustc_ast::expand::autodiff_attrs::{AutoDiffAttrs, DiffActivity, DiffMode};
use rustc_ast::{LitKind, MetaItem, MetaItemInner, attr};
use rustc_attr_data_structures::{
    AttributeKind, InlineAttr, InstructionSetAttr, OptimizeAttr, ReprAttr, UsedBy, find_attr,
};
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LOCAL_CRATE, LocalDefId};
use rustc_hir::weak_lang_items::WEAK_LANG_ITEMS;
use rustc_hir::{self as hir, LangItem, lang_items};
use rustc_middle::middle::codegen_fn_attrs::{
    CodegenFnAttrFlags, CodegenFnAttrs, PatchableFunctionEntry,
};
use rustc_middle::mir::mono::Linkage;
use rustc_middle::query::Providers;
use rustc_middle::span_bug;
use rustc_middle::ty::{self as ty, TyCtxt};
use rustc_session::lint;
use rustc_session::parse::feature_err;
use rustc_span::{Ident, Span, sym};
use rustc_target::spec::SanitizerSet;

use crate::errors;
use crate::errors::NoMangleNameless;
use crate::target_features::{
    check_target_feature_trait_unsafe, check_tied_features, from_target_feature_attr,
};

fn linkage_by_name(tcx: TyCtxt<'_>, def_id: LocalDefId, name: &str) -> Linkage {
    use rustc_middle::mir::mono::Linkage::*;

    // Use the names from src/llvm/docs/LangRef.rst here. Most types are only
    // applicable to variable declarations and may not really make sense for
    // Rust code in the first place but allow them anyway and trust that the
    // user knows what they're doing. Who knows, unanticipated use cases may pop
    // up in the future.
    //
    // ghost, dllimport, dllexport and linkonce_odr_autohide are not supported
    // and don't have to be, LLVM treats them as no-ops.
    match name {
        "available_externally" => AvailableExternally,
        "common" => Common,
        "extern_weak" => ExternalWeak,
        "external" => External,
        "internal" => Internal,
        "linkonce" => LinkOnceAny,
        "linkonce_odr" => LinkOnceODR,
        "weak" => WeakAny,
        "weak_odr" => WeakODR,
        _ => tcx.dcx().span_fatal(tcx.def_span(def_id), "invalid linkage specified"),
    }
}

fn codegen_fn_attrs(tcx: TyCtxt<'_>, did: LocalDefId) -> CodegenFnAttrs {
    if cfg!(debug_assertions) {
        let def_kind = tcx.def_kind(did);
        assert!(
            def_kind.has_codegen_attrs(),
            "unexpected `def_kind` in `codegen_fn_attrs`: {def_kind:?}",
        );
    }

    let attrs = tcx.hir_attrs(tcx.local_def_id_to_hir_id(did));
    let mut codegen_fn_attrs = CodegenFnAttrs::new();
    if tcx.should_inherit_track_caller(did) {
        codegen_fn_attrs.flags |= CodegenFnAttrFlags::TRACK_CALLER;
    }

    // If our rustc version supports autodiff/enzyme, then we call our handler
    // to check for any `#[rustc_autodiff(...)]` attributes.
    if cfg!(llvm_enzyme) {
        let ad = autodiff_attrs(tcx, did.into());
        codegen_fn_attrs.autodiff_item = ad;
    }

    // When `no_builtins` is applied at the crate level, we should add the
    // `no-builtins` attribute to each function to ensure it takes effect in LTO.
    let crate_attrs = tcx.hir_attrs(rustc_hir::CRATE_HIR_ID);
    let no_builtins = attr::contains_name(crate_attrs, sym::no_builtins);
    if no_builtins {
        codegen_fn_attrs.flags |= CodegenFnAttrFlags::NO_BUILTINS;
    }

    let rust_target_features = tcx.rust_target_features(LOCAL_CRATE);

    let mut link_ordinal_span = None;
    let mut no_sanitize_span = None;

    for attr in attrs.iter() {
        // In some cases, attribute are only valid on functions, but it's the `check_attr`
        // pass that check that they aren't used anywhere else, rather this module.
        // In these cases, we bail from performing further checks that are only meaningful for
        // functions (such as calling `fn_sig`, which ICEs if given a non-function). We also
        // report a delayed bug, just in case `check_attr` isn't doing its job.
        let fn_sig = |attr_span| {
            use DefKind::*;

            let def_kind = tcx.def_kind(did);
            if let Fn | AssocFn | Variant | Ctor(..) = def_kind {
                Some(tcx.fn_sig(did))
            } else {
                tcx.dcx()
                    .span_delayed_bug(attr_span, "this attribute can only be applied to functions");
                None
            }
        };

        if let hir::Attribute::Parsed(p) = attr {
            match p {
                AttributeKind::Repr(reprs) => {
                    codegen_fn_attrs.alignment = reprs
                        .iter()
                        .filter_map(
                            |(r, _)| if let ReprAttr::ReprAlign(x) = r { Some(*x) } else { None },
                        )
                        .max();
                }
                AttributeKind::Cold(_) => codegen_fn_attrs.flags |= CodegenFnAttrFlags::COLD,
                AttributeKind::ExportName { name, .. } => {
                    codegen_fn_attrs.export_name = Some(*name);
                }
                AttributeKind::Naked(_) => codegen_fn_attrs.flags |= CodegenFnAttrFlags::NAKED,
                AttributeKind::Align { align, .. } => codegen_fn_attrs.alignment = Some(*align),
                AttributeKind::LinkName { name, .. } => codegen_fn_attrs.link_name = Some(*name),
                AttributeKind::LinkSection { name, .. } => {
                    codegen_fn_attrs.link_section = Some(*name)
                }
                AttributeKind::NoMangle(attr_span) => {
                    if tcx.opt_item_name(did.to_def_id()).is_some() {
                        codegen_fn_attrs.flags |= CodegenFnAttrFlags::NO_MANGLE;
                    } else {
                        tcx.dcx().emit_err(NoMangleNameless {
                            span: *attr_span,
                            definition: format!(
                                "{} {}",
                                tcx.def_descr_article(did.to_def_id()),
                                tcx.def_descr(did.to_def_id())
                            ),
                        });
                    }
                }
                AttributeKind::TrackCaller(attr_span) => {
                    let is_closure = tcx.is_closure_like(did.to_def_id());

                    if !is_closure
                        && let Some(fn_sig) = fn_sig(*attr_span)
                        && fn_sig.skip_binder().abi() != ExternAbi::Rust
                    {
                        tcx.dcx().emit_err(errors::RequiresRustAbi { span: *attr_span });
                    }
                    if is_closure
                        && !tcx.features().closure_track_caller()
                        && !attr_span.allows_unstable(sym::closure_track_caller)
                    {
                        feature_err(
                            &tcx.sess,
                            sym::closure_track_caller,
                            *attr_span,
                            "`#[track_caller]` on closures is currently unstable",
                        )
                        .emit();
                    }
                    codegen_fn_attrs.flags |= CodegenFnAttrFlags::TRACK_CALLER
                }
                AttributeKind::Used { used_by, .. } => match used_by {
                    UsedBy::Compiler => codegen_fn_attrs.flags |= CodegenFnAttrFlags::USED_COMPILER,
                    UsedBy::Linker => codegen_fn_attrs.flags |= CodegenFnAttrFlags::USED_LINKER,
                },
                _ => {}
            }
        }

        let Some(Ident { name, .. }) = attr.ident() else {
            continue;
        };

        match name {
            sym::rustc_allocator => codegen_fn_attrs.flags |= CodegenFnAttrFlags::ALLOCATOR,
            sym::ffi_pure => codegen_fn_attrs.flags |= CodegenFnAttrFlags::FFI_PURE,
            sym::ffi_const => codegen_fn_attrs.flags |= CodegenFnAttrFlags::FFI_CONST,
            sym::rustc_nounwind => codegen_fn_attrs.flags |= CodegenFnAttrFlags::NEVER_UNWIND,
            sym::rustc_reallocator => codegen_fn_attrs.flags |= CodegenFnAttrFlags::REALLOCATOR,
            sym::rustc_deallocator => codegen_fn_attrs.flags |= CodegenFnAttrFlags::DEALLOCATOR,
            sym::rustc_allocator_zeroed => {
                codegen_fn_attrs.flags |= CodegenFnAttrFlags::ALLOCATOR_ZEROED
            }
            sym::rustc_std_internal_symbol => {
                codegen_fn_attrs.flags |= CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL
            }
            sym::thread_local => codegen_fn_attrs.flags |= CodegenFnAttrFlags::THREAD_LOCAL,
            sym::target_feature => {
                let Some(sig) = tcx.hir_node_by_def_id(did).fn_sig() else {
                    tcx.dcx().span_delayed_bug(attr.span(), "target_feature applied to non-fn");
                    continue;
                };
                let safe_target_features =
                    matches!(sig.header.safety, hir::HeaderSafety::SafeTargetFeatures);
                codegen_fn_attrs.safe_target_features = safe_target_features;
                if safe_target_features {
                    if tcx.sess.target.is_like_wasm || tcx.sess.opts.actually_rustdoc {
                        // The `#[target_feature]` attribute is allowed on
                        // WebAssembly targets on all functions. Prior to stabilizing
                        // the `target_feature_11` feature, `#[target_feature]` was
                        // only permitted on unsafe functions because on most targets
                        // execution of instructions that are not supported is
                        // considered undefined behavior. For WebAssembly which is a
                        // 100% safe target at execution time it's not possible to
                        // execute undefined instructions, and even if a future
                        // feature was added in some form for this it would be a
                        // deterministic trap. There is no undefined behavior when
                        // executing WebAssembly so `#[target_feature]` is allowed
                        // on safe functions (but again, only for WebAssembly)
                        //
                        // Note that this is also allowed if `actually_rustdoc` so
                        // if a target is documenting some wasm-specific code then
                        // it's not spuriously denied.
                        //
                        // Now that `#[target_feature]` is permitted on safe functions,
                        // this exception must still exist for allowing the attribute on
                        // `main`, `start`, and other functions that are not usually
                        // allowed.
                    } else {
                        check_target_feature_trait_unsafe(tcx, did, attr.span());
                    }
                }
                from_target_feature_attr(
                    tcx,
                    did,
                    attr,
                    rust_target_features,
                    &mut codegen_fn_attrs.target_features,
                );
            }
            sym::linkage => {
                if let Some(val) = attr.value_str() {
                    let linkage = Some(linkage_by_name(tcx, did, val.as_str()));
                    if tcx.is_foreign_item(did) {
                        codegen_fn_attrs.import_linkage = linkage;

                        if tcx.is_mutable_static(did.into()) {
                            let mut diag = tcx.dcx().struct_span_err(
                                attr.span(),
                                "extern mutable statics are not allowed with `#[linkage]`",
                            );
                            diag.note(
                                "marking the extern static mutable would allow changing which \
                                 symbol the static references rather than make the target of the \
                                 symbol mutable",
                            );
                            diag.emit();
                        }
                    } else {
                        codegen_fn_attrs.linkage = linkage;
                    }
                }
            }
            sym::link_ordinal => {
                link_ordinal_span = Some(attr.span());
                if let ordinal @ Some(_) = check_link_ordinal(tcx, attr) {
                    codegen_fn_attrs.link_ordinal = ordinal;
                }
            }
            sym::no_sanitize => {
                no_sanitize_span = Some(attr.span());
                if let Some(list) = attr.meta_item_list() {
                    for item in list.iter() {
                        match item.name() {
                            Some(sym::address) => {
                                codegen_fn_attrs.no_sanitize |=
                                    SanitizerSet::ADDRESS | SanitizerSet::KERNELADDRESS
                            }
                            Some(sym::cfi) => codegen_fn_attrs.no_sanitize |= SanitizerSet::CFI,
                            Some(sym::kcfi) => codegen_fn_attrs.no_sanitize |= SanitizerSet::KCFI,
                            Some(sym::memory) => {
                                codegen_fn_attrs.no_sanitize |= SanitizerSet::MEMORY
                            }
                            Some(sym::memtag) => {
                                codegen_fn_attrs.no_sanitize |= SanitizerSet::MEMTAG
                            }
                            Some(sym::shadow_call_stack) => {
                                codegen_fn_attrs.no_sanitize |= SanitizerSet::SHADOWCALLSTACK
                            }
                            Some(sym::thread) => {
                                codegen_fn_attrs.no_sanitize |= SanitizerSet::THREAD
                            }
                            Some(sym::hwaddress) => {
                                codegen_fn_attrs.no_sanitize |= SanitizerSet::HWADDRESS
                            }
                            _ => {
                                tcx.dcx().emit_err(errors::InvalidNoSanitize { span: item.span() });
                            }
                        }
                    }
                }
            }
            sym::instruction_set => {
                codegen_fn_attrs.instruction_set =
                    attr.meta_item_list().and_then(|l| match &l[..] {
                        [MetaItemInner::MetaItem(set)] => {
                            let segments =
                                set.path.segments.iter().map(|x| x.ident.name).collect::<Vec<_>>();
                            match segments.as_slice() {
                                [sym::arm, sym::a32 | sym::t32]
                                    if !tcx.sess.target.has_thumb_interworking =>
                                {
                                    tcx.dcx().emit_err(errors::UnsupportedInstructionSet {
                                        span: attr.span(),
                                    });
                                    None
                                }
                                [sym::arm, sym::a32] => Some(InstructionSetAttr::ArmA32),
                                [sym::arm, sym::t32] => Some(InstructionSetAttr::ArmT32),
                                _ => {
                                    tcx.dcx().emit_err(errors::InvalidInstructionSet {
                                        span: attr.span(),
                                    });
                                    None
                                }
                            }
                        }
                        [] => {
                            tcx.dcx().emit_err(errors::BareInstructionSet { span: attr.span() });
                            None
                        }
                        _ => {
                            tcx.dcx()
                                .emit_err(errors::MultipleInstructionSet { span: attr.span() });
                            None
                        }
                    })
            }
            sym::patchable_function_entry => {
                codegen_fn_attrs.patchable_function_entry = attr.meta_item_list().and_then(|l| {
                    let mut prefix = None;
                    let mut entry = None;
                    for item in l {
                        let Some(meta_item) = item.meta_item() else {
                            tcx.dcx().emit_err(errors::ExpectedNameValuePair { span: item.span() });
                            continue;
                        };

                        let Some(name_value_lit) = meta_item.name_value_literal() else {
                            tcx.dcx().emit_err(errors::ExpectedNameValuePair { span: item.span() });
                            continue;
                        };

                        let attrib_to_write = match meta_item.name() {
                            Some(sym::prefix_nops) => &mut prefix,
                            Some(sym::entry_nops) => &mut entry,
                            _ => {
                                tcx.dcx().emit_err(errors::UnexpectedParameterName {
                                    span: item.span(),
                                    prefix_nops: sym::prefix_nops,
                                    entry_nops: sym::entry_nops,
                                });
                                continue;
                            }
                        };

                        let rustc_ast::LitKind::Int(val, _) = name_value_lit.kind else {
                            tcx.dcx().emit_err(errors::InvalidLiteralValue {
                                span: name_value_lit.span,
                            });
                            continue;
                        };

                        let Ok(val) = val.get().try_into() else {
                            tcx.dcx()
                                .emit_err(errors::OutOfRangeInteger { span: name_value_lit.span });
                            continue;
                        };

                        *attrib_to_write = Some(val);
                    }

                    if let (None, None) = (prefix, entry) {
                        tcx.dcx().span_err(attr.span(), "must specify at least one parameter");
                    }

                    Some(PatchableFunctionEntry::from_prefix_and_entry(
                        prefix.unwrap_or(0),
                        entry.unwrap_or(0),
                    ))
                })
            }
            _ => {}
        }
    }

    // Apply the minimum function alignment here. This ensures that a function's alignment is
    // determined by the `-C` flags of the crate it is defined in, not the `-C` flags of the crate
    // it happens to be codegen'd (or const-eval'd) in.
    codegen_fn_attrs.alignment =
        Ord::max(codegen_fn_attrs.alignment, tcx.sess.opts.unstable_opts.min_function_alignment);

    // On trait methods, inherit the `#[align]` of the trait's method prototype.
    codegen_fn_attrs.alignment = Ord::max(codegen_fn_attrs.alignment, tcx.inherited_align(did));

    let inline_span;
    (codegen_fn_attrs.inline, inline_span) = if let Some((inline_attr, span)) =
        find_attr!(attrs, AttributeKind::Inline(i, span) => (*i, *span))
    {
        (inline_attr, Some(span))
    } else {
        (InlineAttr::None, None)
    };

    // naked function MUST NOT be inlined! This attribute is required for the rust compiler itself,
    // but not for the code generation backend because at that point the naked function will just be
    // a declaration, with a definition provided in global assembly.
    if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::NAKED) {
        codegen_fn_attrs.inline = InlineAttr::Never;
    }

    codegen_fn_attrs.optimize =
        find_attr!(attrs, AttributeKind::Optimize(i, _) => *i).unwrap_or(OptimizeAttr::Default);

    // #73631: closures inherit `#[target_feature]` annotations
    //
    // If this closure is marked `#[inline(always)]`, simply skip adding `#[target_feature]`.
    //
    // At this point, `unsafe` has already been checked and `#[target_feature]` only affects codegen.
    // Due to LLVM limitations, emitting both `#[inline(always)]` and `#[target_feature]` is *unsound*:
    // the function may be inlined into a caller with fewer target features. Also see
    // <https://github.com/rust-lang/rust/issues/116573>.
    //
    // Using `#[inline(always)]` implies that this closure will most likely be inlined into
    // its parent function, which effectively inherits the features anyway. Boxing this closure
    // would result in this closure being compiled without the inherited target features, but this
    // is probably a poor usage of `#[inline(always)]` and easily avoided by not using the attribute.
    if tcx.is_closure_like(did.to_def_id()) && codegen_fn_attrs.inline != InlineAttr::Always {
        let owner_id = tcx.parent(did.to_def_id());
        if tcx.def_kind(owner_id).has_codegen_attrs() {
            codegen_fn_attrs
                .target_features
                .extend(tcx.codegen_fn_attrs(owner_id).target_features.iter().copied());
        }
    }

    // If a function uses `#[target_feature]` it can't be inlined into general
    // purpose functions as they wouldn't have the right target features
    // enabled. For that reason we also forbid `#[inline(always)]` as it can't be
    // respected.
    //
    // `#[rustc_force_inline]` doesn't need to be prohibited here, only
    // `#[inline(always)]`, as forced inlining is implemented entirely within
    // rustc (and so the MIR inliner can do any necessary checks for compatible target
    // features).
    //
    // This sidesteps the LLVM blockers in enabling `target_features` +
    // `inline(always)` to be used together (see rust-lang/rust#116573 and
    // llvm/llvm-project#70563).
    if !codegen_fn_attrs.target_features.is_empty()
        && matches!(codegen_fn_attrs.inline, InlineAttr::Always)
        && let Some(span) = inline_span
    {
        tcx.dcx().span_err(span, "cannot use `#[inline(always)]` with `#[target_feature]`");
    }

    if !codegen_fn_attrs.no_sanitize.is_empty()
        && codegen_fn_attrs.inline.always()
        && let (Some(no_sanitize_span), Some(inline_span)) = (no_sanitize_span, inline_span)
    {
        let hir_id = tcx.local_def_id_to_hir_id(did);
        tcx.node_span_lint(lint::builtin::INLINE_NO_SANITIZE, hir_id, no_sanitize_span, |lint| {
            lint.primary_message("`no_sanitize` will have no effect after inlining");
            lint.span_note(inline_span, "inlining requested here");
        })
    }

    // Weak lang items have the same semantics as "std internal" symbols in the
    // sense that they're preserved through all our LTO passes and only
    // strippable by the linker.
    //
    // Additionally weak lang items have predetermined symbol names.
    if let Some((name, _)) = lang_items::extract(attrs)
        && let Some(lang_item) = LangItem::from_name(name)
    {
        if WEAK_LANG_ITEMS.contains(&lang_item) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL;
        }
        if let Some(link_name) = lang_item.link_name() {
            codegen_fn_attrs.export_name = Some(link_name);
            codegen_fn_attrs.link_name = Some(link_name);
        }
    }
    check_link_name_xor_ordinal(tcx, &codegen_fn_attrs, link_ordinal_span);

    if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL)
        && codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::NO_MANGLE)
    {
        let no_mangle_span =
            find_attr!(attrs, AttributeKind::NoMangle(no_mangle_span) => *no_mangle_span)
                .unwrap_or_default();
        let lang_item =
            lang_items::extract(attrs).map_or(None, |(name, _span)| LangItem::from_name(name));
        let mut err = tcx
            .dcx()
            .struct_span_err(
                no_mangle_span,
                "`#[no_mangle]` cannot be used on internal language items",
            )
            .with_note("Rustc requires this item to have a specific mangled name.")
            .with_span_label(tcx.def_span(did), "should be the internal language item");
        if let Some(lang_item) = lang_item {
            if let Some(link_name) = lang_item.link_name() {
                err = err
                    .with_note("If you are trying to prevent mangling to ease debugging, many")
                    .with_note(format!(
                        "debuggers support a command such as `rbreak {link_name}` to"
                    ))
                    .with_note(format!(
                        "match `.*{link_name}.*` instead of `break {link_name}` on a specific name"
                    ))
            }
        }
        err.emit();
    }

    // Any linkage to LLVM intrinsics for now forcibly marks them all as never
    // unwinds since LLVM sometimes can't handle codegen which `invoke`s
    // intrinsic functions.
    if let Some(name) = &codegen_fn_attrs.link_name
        && name.as_str().starts_with("llvm.")
    {
        codegen_fn_attrs.flags |= CodegenFnAttrFlags::NEVER_UNWIND;
    }

    if let Some(features) = check_tied_features(
        tcx.sess,
        &codegen_fn_attrs
            .target_features
            .iter()
            .map(|features| (features.name.as_str(), true))
            .collect(),
    ) {
        let span = tcx
            .get_attrs(did, sym::target_feature)
            .next()
            .map_or_else(|| tcx.def_span(did), |a| a.span());
        tcx.dcx()
            .create_err(errors::TargetFeatureDisableOrEnable {
                features,
                span: Some(span),
                missing_features: Some(errors::MissingFeatures),
            })
            .emit();
    }

    codegen_fn_attrs
}

/// If the provided DefId is a method in a trait impl, return the DefId of the method prototype.
fn opt_trait_item(tcx: TyCtxt<'_>, def_id: DefId) -> Option<DefId> {
    let impl_item = tcx.opt_associated_item(def_id)?;
    match impl_item.container {
        ty::AssocItemContainer::Impl => impl_item.trait_item_def_id,
        _ => None,
    }
}

/// Checks if the provided DefId is a method in a trait impl for a trait which has track_caller
/// applied to the method prototype.
fn should_inherit_track_caller(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    let Some(trait_item) = opt_trait_item(tcx, def_id) else { return false };
    tcx.codegen_fn_attrs(trait_item).flags.intersects(CodegenFnAttrFlags::TRACK_CALLER)
}

/// If the provided DefId is a method in a trait impl, return the value of the `#[align]`
/// attribute on the method prototype (if any).
fn inherited_align<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId) -> Option<Align> {
    tcx.codegen_fn_attrs(opt_trait_item(tcx, def_id)?).alignment
}

fn check_link_ordinal(tcx: TyCtxt<'_>, attr: &hir::Attribute) -> Option<u16> {
    use rustc_ast::{LitIntType, LitKind, MetaItemLit};
    let meta_item_list = attr.meta_item_list()?;
    let [sole_meta_list] = &meta_item_list[..] else {
        tcx.dcx().emit_err(errors::InvalidLinkOrdinalNargs { span: attr.span() });
        return None;
    };
    if let Some(MetaItemLit { kind: LitKind::Int(ordinal, LitIntType::Unsuffixed), .. }) =
        sole_meta_list.lit()
    {
        // According to the table at
        // https://docs.microsoft.com/en-us/windows/win32/debug/pe-format#import-header, the
        // ordinal must fit into 16 bits. Similarly, the Ordinal field in COFFShortExport (defined
        // in llvm/include/llvm/Object/COFFImportFile.h), which we use to communicate import
        // information to LLVM for `#[link(kind = "raw-dylib"_])`, is also defined to be uint16_t.
        //
        // FIXME: should we allow an ordinal of 0?  The MSVC toolchain has inconsistent support for
        // this: both LINK.EXE and LIB.EXE signal errors and abort when given a .DEF file that
        // specifies a zero ordinal. However, llvm-dlltool is perfectly happy to generate an import
        // library for such a .DEF file, and MSVC's LINK.EXE is also perfectly happy to consume an
        // import library produced by LLVM with an ordinal of 0, and it generates an .EXE.  (I
        // don't know yet if the resulting EXE runs, as I haven't yet built the necessary DLL --
        // see earlier comment about LINK.EXE failing.)
        if *ordinal <= u16::MAX as u128 {
            Some(ordinal.get() as u16)
        } else {
            let msg = format!("ordinal value in `link_ordinal` is too large: `{ordinal}`");
            tcx.dcx()
                .struct_span_err(attr.span(), msg)
                .with_note("the value may not exceed `u16::MAX`")
                .emit();
            None
        }
    } else {
        tcx.dcx().emit_err(errors::InvalidLinkOrdinalFormat { span: attr.span() });
        None
    }
}

fn check_link_name_xor_ordinal(
    tcx: TyCtxt<'_>,
    codegen_fn_attrs: &CodegenFnAttrs,
    inline_span: Option<Span>,
) {
    if codegen_fn_attrs.link_name.is_none() || codegen_fn_attrs.link_ordinal.is_none() {
        return;
    }
    let msg = "cannot use `#[link_name]` with `#[link_ordinal]`";
    if let Some(span) = inline_span {
        tcx.dcx().span_err(span, msg);
    } else {
        tcx.dcx().err(msg);
    }
}

/// We now check the #\[rustc_autodiff\] attributes which we generated from the #[autodiff(...)]
/// macros. There are two forms. The pure one without args to mark primal functions (the functions
/// being differentiated). The other form is #[rustc_autodiff(Mode, ActivityList)] on top of the
/// placeholder functions. We wrote the rustc_autodiff attributes ourself, so this should never
/// panic, unless we introduced a bug when parsing the autodiff macro.
fn autodiff_attrs(tcx: TyCtxt<'_>, id: DefId) -> Option<AutoDiffAttrs> {
    let attrs = tcx.get_attrs(id, sym::rustc_autodiff);

    let attrs = attrs.filter(|attr| attr.has_name(sym::rustc_autodiff)).collect::<Vec<_>>();

    // check for exactly one autodiff attribute on placeholder functions.
    // There should only be one, since we generate a new placeholder per ad macro.
    let attr = match &attrs[..] {
        [] => return None,
        [attr] => attr,
        _ => {
            span_bug!(attrs[1].span(), "cg_ssa: rustc_autodiff should only exist once per source");
        }
    };

    let list = attr.meta_item_list().unwrap_or_default();

    // empty autodiff attribute macros (i.e. `#[autodiff]`) are used to mark source functions
    if list.is_empty() {
        return Some(AutoDiffAttrs::source());
    }

    let [mode, width_meta, input_activities @ .., ret_activity] = &list[..] else {
        span_bug!(attr.span(), "rustc_autodiff attribute must contain mode, width and activities");
    };
    let mode = if let MetaItemInner::MetaItem(MetaItem { path: p1, .. }) = mode {
        p1.segments.first().unwrap().ident
    } else {
        span_bug!(attr.span(), "rustc_autodiff attribute must contain mode");
    };

    // parse mode
    let mode = match mode.as_str() {
        "Forward" => DiffMode::Forward,
        "Reverse" => DiffMode::Reverse,
        _ => {
            span_bug!(mode.span, "rustc_autodiff attribute contains invalid mode");
        }
    };

    let width: u32 = match width_meta {
        MetaItemInner::MetaItem(MetaItem { path: p1, .. }) => {
            let w = p1.segments.first().unwrap().ident;
            match w.as_str().parse() {
                Ok(val) => val,
                Err(_) => {
                    span_bug!(w.span, "rustc_autodiff width should fit u32");
                }
            }
        }
        MetaItemInner::Lit(lit) => {
            if let LitKind::Int(val, _) = lit.kind {
                match val.get().try_into() {
                    Ok(val) => val,
                    Err(_) => {
                        span_bug!(lit.span, "rustc_autodiff width should fit u32");
                    }
                }
            } else {
                span_bug!(lit.span, "rustc_autodiff width should be an integer");
            }
        }
    };

    // First read the ret symbol from the attribute
    let ret_symbol = if let MetaItemInner::MetaItem(MetaItem { path: p1, .. }) = ret_activity {
        p1.segments.first().unwrap().ident
    } else {
        span_bug!(attr.span(), "rustc_autodiff attribute must contain the return activity");
    };

    // Then parse it into an actual DiffActivity
    let Ok(ret_activity) = DiffActivity::from_str(ret_symbol.as_str()) else {
        span_bug!(ret_symbol.span, "invalid return activity");
    };

    // Now parse all the intermediate (input) activities
    let mut arg_activities: Vec<DiffActivity> = vec![];
    for arg in input_activities {
        let arg_symbol = if let MetaItemInner::MetaItem(MetaItem { path: p2, .. }) = arg {
            match p2.segments.first() {
                Some(x) => x.ident,
                None => {
                    span_bug!(
                        arg.span(),
                        "rustc_autodiff attribute must contain the input activity"
                    );
                }
            }
        } else {
            span_bug!(arg.span(), "rustc_autodiff attribute must contain the input activity");
        };

        match DiffActivity::from_str(arg_symbol.as_str()) {
            Ok(arg_activity) => arg_activities.push(arg_activity),
            Err(_) => {
                span_bug!(arg_symbol.span, "invalid input activity");
            }
        }
    }

    Some(AutoDiffAttrs { mode, width, ret_activity, input_activity: arg_activities })
}

pub(crate) fn provide(providers: &mut Providers) {
    *providers =
        Providers { codegen_fn_attrs, should_inherit_track_caller, inherited_align, ..*providers };
}
