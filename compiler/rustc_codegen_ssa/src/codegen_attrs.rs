use std::str::FromStr;

use rustc_abi::{Align, ExternAbi};
use rustc_ast::expand::autodiff_attrs::{AutoDiffAttrs, DiffActivity, DiffMode};
use rustc_ast::{LitKind, MetaItem, MetaItemInner, attr};
use rustc_hir::attrs::{AttributeKind, InlineAttr, InstructionSetAttr, UsedBy};
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LOCAL_CRATE, LocalDefId};
use rustc_hir::weak_lang_items::WEAK_LANG_ITEMS;
use rustc_hir::{self as hir, Attribute, LangItem, find_attr, lang_items};
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

/// In some cases, attributes are only valid on functions, but it's the `check_attr`
/// pass that checks that they aren't used anywhere else, rather than this module.
/// In these cases, we bail from performing further checks that are only meaningful for
/// functions (such as calling `fn_sig`, which ICEs if given a non-function). We also
/// report a delayed bug, just in case `check_attr` isn't doing its job.
fn try_fn_sig<'tcx>(
    tcx: TyCtxt<'tcx>,
    did: LocalDefId,
    attr_span: Span,
) -> Option<ty::EarlyBinder<'tcx, ty::PolyFnSig<'tcx>>> {
    use DefKind::*;

    let def_kind = tcx.def_kind(did);
    if let Fn | AssocFn | Variant | Ctor(..) = def_kind {
        Some(tcx.fn_sig(did))
    } else {
        tcx.dcx().span_delayed_bug(attr_span, "this attribute can only be applied to functions");
        None
    }
}

// FIXME(jdonszelmann): remove when instruction_set becomes a parsed attr
fn parse_instruction_set_attr(tcx: TyCtxt<'_>, attr: &Attribute) -> Option<InstructionSetAttr> {
    let list = attr.meta_item_list()?;

    match &list[..] {
        [MetaItemInner::MetaItem(set)] => {
            let segments = set.path.segments.iter().map(|x| x.ident.name).collect::<Vec<_>>();
            match segments.as_slice() {
                [sym::arm, sym::a32 | sym::t32] if !tcx.sess.target.has_thumb_interworking => {
                    tcx.dcx().emit_err(errors::UnsupportedInstructionSet { span: attr.span() });
                    None
                }
                [sym::arm, sym::a32] => Some(InstructionSetAttr::ArmA32),
                [sym::arm, sym::t32] => Some(InstructionSetAttr::ArmT32),
                _ => {
                    tcx.dcx().emit_err(errors::InvalidInstructionSet { span: attr.span() });
                    None
                }
            }
        }
        [] => {
            tcx.dcx().emit_err(errors::BareInstructionSet { span: attr.span() });
            None
        }
        _ => {
            tcx.dcx().emit_err(errors::MultipleInstructionSet { span: attr.span() });
            None
        }
    }
}

// FIXME(jdonszelmann): remove when linkage becomes a parsed attr
fn parse_linkage_attr(tcx: TyCtxt<'_>, did: LocalDefId, attr: &Attribute) -> Option<Linkage> {
    let val = attr.value_str()?;
    let linkage = linkage_by_name(tcx, did, val.as_str());
    Some(linkage)
}

// FIXME(jdonszelmann): remove when no_sanitize becomes a parsed attr
fn parse_no_sanitize_attr(tcx: TyCtxt<'_>, attr: &Attribute) -> Option<SanitizerSet> {
    let list = attr.meta_item_list()?;
    let mut sanitizer_set = SanitizerSet::empty();

    for item in list.iter() {
        match item.name() {
            Some(sym::address) => {
                sanitizer_set |= SanitizerSet::ADDRESS | SanitizerSet::KERNELADDRESS
            }
            Some(sym::cfi) => sanitizer_set |= SanitizerSet::CFI,
            Some(sym::kcfi) => sanitizer_set |= SanitizerSet::KCFI,
            Some(sym::memory) => sanitizer_set |= SanitizerSet::MEMORY,
            Some(sym::memtag) => sanitizer_set |= SanitizerSet::MEMTAG,
            Some(sym::shadow_call_stack) => sanitizer_set |= SanitizerSet::SHADOWCALLSTACK,
            Some(sym::thread) => sanitizer_set |= SanitizerSet::THREAD,
            Some(sym::hwaddress) => sanitizer_set |= SanitizerSet::HWADDRESS,
            _ => {
                tcx.dcx().emit_err(errors::InvalidNoSanitize { span: item.span() });
            }
        }
    }

    Some(sanitizer_set)
}

// FIXME(jdonszelmann): remove when patchable_function_entry becomes a parsed attr
fn parse_patchable_function_entry(
    tcx: TyCtxt<'_>,
    attr: &Attribute,
) -> Option<PatchableFunctionEntry> {
    attr.meta_item_list().and_then(|l| {
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
                tcx.dcx().emit_err(errors::InvalidLiteralValue { span: name_value_lit.span });
                continue;
            };

            let Ok(val) = val.get().try_into() else {
                tcx.dcx().emit_err(errors::OutOfRangeInteger { span: name_value_lit.span });
                continue;
            };

            *attrib_to_write = Some(val);
        }

        if let (None, None) = (prefix, entry) {
            tcx.dcx().span_err(attr.span(), "must specify at least one parameter");
        }

        Some(PatchableFunctionEntry::from_prefix_and_entry(prefix.unwrap_or(0), entry.unwrap_or(0)))
    })
}

/// Spans that are collected when processing built-in attributes,
/// that are useful for emitting diagnostics later.
#[derive(Default)]
struct InterestingAttributeDiagnosticSpans {
    link_ordinal: Option<Span>,
    no_sanitize: Option<Span>,
    inline: Option<Span>,
    no_mangle: Option<Span>,
}

/// Process the builtin attrs ([`hir::Attribute`]) on the item.
/// Many of them directly translate to codegen attrs.
fn process_builtin_attrs(
    tcx: TyCtxt<'_>,
    did: LocalDefId,
    attrs: &[Attribute],
    codegen_fn_attrs: &mut CodegenFnAttrs,
) -> InterestingAttributeDiagnosticSpans {
    let mut interesting_spans = InterestingAttributeDiagnosticSpans::default();
    let rust_target_features = tcx.rust_target_features(LOCAL_CRATE);

    // If our rustc version supports autodiff/enzyme, then we call our handler
    // to check for any `#[rustc_autodiff(...)]` attributes.
    // FIXME(jdonszelmann): merge with loop below
    if cfg!(llvm_enzyme) {
        let ad = autodiff_attrs(tcx, did.into());
        codegen_fn_attrs.autodiff_item = ad;
    }

    for attr in attrs.iter() {
        if let hir::Attribute::Parsed(p) = attr {
            match p {
                AttributeKind::Cold(_) => codegen_fn_attrs.flags |= CodegenFnAttrFlags::COLD,
                AttributeKind::ExportName { name, .. } => {
                    codegen_fn_attrs.export_name = Some(*name)
                }
                AttributeKind::Inline(inline, span) => {
                    codegen_fn_attrs.inline = *inline;
                    interesting_spans.inline = Some(*span);
                }
                AttributeKind::Naked(_) => codegen_fn_attrs.flags |= CodegenFnAttrFlags::NAKED,
                AttributeKind::Align { align, .. } => codegen_fn_attrs.alignment = Some(*align),
                AttributeKind::LinkName { name, .. } => codegen_fn_attrs.link_name = Some(*name),
                AttributeKind::LinkOrdinal { ordinal, span } => {
                    codegen_fn_attrs.link_ordinal = Some(*ordinal);
                    interesting_spans.link_ordinal = Some(*span);
                }
                AttributeKind::LinkSection { name, .. } => {
                    codegen_fn_attrs.link_section = Some(*name)
                }
                AttributeKind::NoMangle(attr_span) => {
                    interesting_spans.no_mangle = Some(*attr_span);
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
                AttributeKind::Optimize(optimize, _) => codegen_fn_attrs.optimize = *optimize,
                AttributeKind::TargetFeature(features, attr_span) => {
                    let Some(sig) = tcx.hir_node_by_def_id(did).fn_sig() else {
                        tcx.dcx().span_delayed_bug(*attr_span, "target_feature applied to non-fn");
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
                            check_target_feature_trait_unsafe(tcx, did, *attr_span);
                        }
                    }
                    from_target_feature_attr(
                        tcx,
                        did,
                        features,
                        rust_target_features,
                        &mut codegen_fn_attrs.target_features,
                    );
                }
                AttributeKind::TrackCaller(attr_span) => {
                    let is_closure = tcx.is_closure_like(did.to_def_id());

                    if !is_closure
                        && let Some(fn_sig) = try_fn_sig(tcx, did, *attr_span)
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
                AttributeKind::FfiConst(_) => {
                    codegen_fn_attrs.flags |= CodegenFnAttrFlags::FFI_CONST
                }
                AttributeKind::FfiPure(_) => codegen_fn_attrs.flags |= CodegenFnAttrFlags::FFI_PURE,
                AttributeKind::StdInternalSymbol(_) => {
                    codegen_fn_attrs.flags |= CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL
                }
                _ => {}
            }
        }

        let Some(Ident { name, .. }) = attr.ident() else {
            continue;
        };

        match name {
            sym::rustc_allocator => codegen_fn_attrs.flags |= CodegenFnAttrFlags::ALLOCATOR,
            sym::rustc_nounwind => codegen_fn_attrs.flags |= CodegenFnAttrFlags::NEVER_UNWIND,
            sym::rustc_reallocator => codegen_fn_attrs.flags |= CodegenFnAttrFlags::REALLOCATOR,
            sym::rustc_deallocator => codegen_fn_attrs.flags |= CodegenFnAttrFlags::DEALLOCATOR,
            sym::rustc_allocator_zeroed => {
                codegen_fn_attrs.flags |= CodegenFnAttrFlags::ALLOCATOR_ZEROED
            }
            sym::thread_local => codegen_fn_attrs.flags |= CodegenFnAttrFlags::THREAD_LOCAL,
            sym::linkage => {
                let linkage = parse_linkage_attr(tcx, did, attr);

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
            sym::no_sanitize => {
                interesting_spans.no_sanitize = Some(attr.span());
                codegen_fn_attrs.no_sanitize |=
                    parse_no_sanitize_attr(tcx, attr).unwrap_or_default();
            }
            sym::instruction_set => {
                codegen_fn_attrs.instruction_set = parse_instruction_set_attr(tcx, attr)
            }
            sym::patchable_function_entry => {
                codegen_fn_attrs.patchable_function_entry =
                    parse_patchable_function_entry(tcx, attr);
            }
            _ => {}
        }
    }

    interesting_spans
}

/// Applies overrides for codegen fn attrs. These often have a specific reason why they're necessary.
/// Please comment why when adding a new one!
fn apply_overrides(tcx: TyCtxt<'_>, did: LocalDefId, codegen_fn_attrs: &mut CodegenFnAttrs) {
    // Apply the minimum function alignment here. This ensures that a function's alignment is
    // determined by the `-C` flags of the crate it is defined in, not the `-C` flags of the crate
    // it happens to be codegen'd (or const-eval'd) in.
    codegen_fn_attrs.alignment =
        Ord::max(codegen_fn_attrs.alignment, tcx.sess.opts.unstable_opts.min_function_alignment);

    // On trait methods, inherit the `#[align]` of the trait's method prototype.
    codegen_fn_attrs.alignment = Ord::max(codegen_fn_attrs.alignment, tcx.inherited_align(did));

    // naked function MUST NOT be inlined! This attribute is required for the rust compiler itself,
    // but not for the code generation backend because at that point the naked function will just be
    // a declaration, with a definition provided in global assembly.
    if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::NAKED) {
        codegen_fn_attrs.inline = InlineAttr::Never;
    }

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

    // When `no_builtins` is applied at the crate level, we should add the
    // `no-builtins` attribute to each function to ensure it takes effect in LTO.
    let crate_attrs = tcx.hir_attrs(rustc_hir::CRATE_HIR_ID);
    let no_builtins = attr::contains_name(crate_attrs, sym::no_builtins);
    if no_builtins {
        codegen_fn_attrs.flags |= CodegenFnAttrFlags::NO_BUILTINS;
    }

    // inherit track-caller properly
    if tcx.should_inherit_track_caller(did) {
        codegen_fn_attrs.flags |= CodegenFnAttrFlags::TRACK_CALLER;
    }

    // Foreign items by default use no mangling for their symbol name.
    if tcx.is_foreign_item(did) {
        // There's a few exceptions to this rule though:
        if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL) {
            // * `#[rustc_std_internal_symbol]` mangles the symbol name in a special way
            //   both for exports and imports through foreign items. This is handled further,
            //   during symbol mangling logic.
        } else if codegen_fn_attrs.link_name.is_some() {
            // * This can be overridden with the `#[link_name]` attribute
        } else {
            // NOTE: there's one more exception that we cannot apply here. On wasm,
            // some items cannot be `no_mangle`.
            // However, we don't have enough information here to determine that.
            // As such, no_mangle foreign items on wasm that have the same defid as some
            // import will *still* be mangled despite this.
            //
            // if none of the exceptions apply; apply no_mangle
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::NO_MANGLE;
        }
    }
}

fn check_result(
    tcx: TyCtxt<'_>,
    did: LocalDefId,
    interesting_spans: InterestingAttributeDiagnosticSpans,
    codegen_fn_attrs: &CodegenFnAttrs,
) {
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
        && let Some(span) = interesting_spans.inline
    {
        tcx.dcx().span_err(span, "cannot use `#[inline(always)]` with `#[target_feature]`");
    }

    // warn that inline has no effect when no_sanitize is present
    if !codegen_fn_attrs.no_sanitize.is_empty()
        && codegen_fn_attrs.inline.always()
        && let (Some(no_sanitize_span), Some(inline_span)) =
            (interesting_spans.no_sanitize, interesting_spans.inline)
    {
        let hir_id = tcx.local_def_id_to_hir_id(did);
        tcx.node_span_lint(lint::builtin::INLINE_NO_SANITIZE, hir_id, no_sanitize_span, |lint| {
            lint.primary_message("`no_sanitize` will have no effect after inlining");
            lint.span_note(inline_span, "inlining requested here");
        })
    }

    // error when specifying link_name together with link_ordinal
    if let Some(_) = codegen_fn_attrs.link_name
        && let Some(_) = codegen_fn_attrs.link_ordinal
    {
        let msg = "cannot use `#[link_name]` with `#[link_ordinal]`";
        if let Some(span) = interesting_spans.link_ordinal {
            tcx.dcx().span_err(span, msg);
        } else {
            tcx.dcx().err(msg);
        }
    }

    if let Some(features) = check_tied_features(
        tcx.sess,
        &codegen_fn_attrs
            .target_features
            .iter()
            .map(|features| (features.name.as_str(), true))
            .collect(),
    ) {
        let span =
            find_attr!(tcx.get_all_attrs(did), AttributeKind::TargetFeature(_, span) => *span)
                .unwrap_or_else(|| tcx.def_span(did));

        tcx.dcx()
            .create_err(errors::TargetFeatureDisableOrEnable {
                features,
                span: Some(span),
                missing_features: Some(errors::MissingFeatures),
            })
            .emit();
    }
}

fn handle_lang_items(
    tcx: TyCtxt<'_>,
    did: LocalDefId,
    interesting_spans: &InterestingAttributeDiagnosticSpans,
    attrs: &[Attribute],
    codegen_fn_attrs: &mut CodegenFnAttrs,
) {
    let lang_item = lang_items::extract(attrs).and_then(|(name, _)| LangItem::from_name(name));

    // Weak lang items have the same semantics as "std internal" symbols in the
    // sense that they're preserved through all our LTO passes and only
    // strippable by the linker.
    //
    // Additionally weak lang items have predetermined symbol names.
    if let Some(lang_item) = lang_item {
        if WEAK_LANG_ITEMS.contains(&lang_item) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL;
        }
        if let Some(link_name) = lang_item.link_name() {
            codegen_fn_attrs.export_name = Some(link_name);
            codegen_fn_attrs.link_name = Some(link_name);
        }
    }

    // error when using no_mangle on a lang item item
    if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL)
        && codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::NO_MANGLE)
    {
        let mut err = tcx
            .dcx()
            .struct_span_err(
                interesting_spans.no_mangle.unwrap_or_default(),
                "`#[no_mangle]` cannot be used on internal language items",
            )
            .with_note("Rustc requires this item to have a specific mangled name.")
            .with_span_label(tcx.def_span(did), "should be the internal language item");
        if let Some(lang_item) = lang_item
            && let Some(link_name) = lang_item.link_name()
        {
            err = err
                .with_note("If you are trying to prevent mangling to ease debugging, many")
                .with_note(format!("debuggers support a command such as `rbreak {link_name}` to"))
                .with_note(format!(
                    "match `.*{link_name}.*` instead of `break {link_name}` on a specific name"
                ))
        }
        err.emit();
    }
}

/// Generate the [`CodegenFnAttrs`] for an item (identified by the [`LocalDefId`]).
///
/// This happens in 4 stages:
/// - apply built-in attributes that directly translate to codegen attributes.
/// - handle lang items. These have special codegen attrs applied to them.
/// - apply overrides, like minimum requirements for alignment and other settings that don't rely directly the built-in attrs on the item.
///   overrides come after applying built-in attributes since they may only apply when certain attributes were already set in the stage before.
/// - check that the result is valid. There's various ways in which this may not be the case, such as certain combinations of attrs.
fn codegen_fn_attrs(tcx: TyCtxt<'_>, did: LocalDefId) -> CodegenFnAttrs {
    if cfg!(debug_assertions) {
        let def_kind = tcx.def_kind(did);
        assert!(
            def_kind.has_codegen_attrs(),
            "unexpected `def_kind` in `codegen_fn_attrs`: {def_kind:?}",
        );
    }

    let mut codegen_fn_attrs = CodegenFnAttrs::new();
    let attrs = tcx.hir_attrs(tcx.local_def_id_to_hir_id(did));

    let interesting_spans = process_builtin_attrs(tcx, did, attrs, &mut codegen_fn_attrs);
    handle_lang_items(tcx, did, &interesting_spans, attrs, &mut codegen_fn_attrs);
    apply_overrides(tcx, did, &mut codegen_fn_attrs);
    check_result(tcx, did, interesting_spans, &codegen_fn_attrs);

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

/// We now check the #\[rustc_autodiff\] attributes which we generated from the #[autodiff(...)]
/// macros. There are two forms. The pure one without args to mark primal functions (the functions
/// being differentiated). The other form is #[rustc_autodiff(Mode, ActivityList)] on top of the
/// placeholder functions. We wrote the rustc_autodiff attributes ourself, so this should never
/// panic, unless we introduced a bug when parsing the autodiff macro.
//FIXME(jdonszelmann): put in the main loop. No need to have two..... :/ Let's do that when we make autodiff parsed.
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
