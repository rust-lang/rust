// FIXME(jdonszelmann): should become rustc_attr_validation
//! This module implements some validity checks for attributes.
//! In particular it verifies that `#[inline]` and `#[repr]` attributes are
//! attached to items that actually support them and if there are
//! conflicts between multiple such attributes attached to the same
//! item.

use std::cell::Cell;
use std::slice;

use rustc_abi::ExternAbi;
use rustc_ast::{AttrStyle, MetaItemKind, ast};
use rustc_attr_parsing::AttributeParser;
use rustc_data_structures::thin_vec::ThinVec;
use rustc_data_structures::unord::UnordMap;
use rustc_errors::{DiagCtxtHandle, IntoDiagArg, MultiSpan, msg};
use rustc_feature::BUILTIN_ATTRIBUTE_MAP;
use rustc_hir::attrs::diagnostic::Directive;
use rustc_hir::attrs::{
    AttributeKind, DocAttribute, DocInline, EiiDecl, EiiImpl, EiiImplResolution, InlineAttr,
    OptimizeAttr, ReprAttr,
};
use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalModId;
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{
    self as hir, Attribute, CRATE_HIR_ID, Constness, FnSig, ForeignItem, GenericParam,
    GenericParamKind, HirId, Item, ItemKind, MethodKind, Node, ParamName, Target, TraitItem,
    find_attr,
};
use rustc_macros::Diagnostic;
use rustc_middle::hir::nested_filter;
use rustc_middle::middle::resolve_bound_vars::ObjectLifetimeDefault;
use rustc_middle::query::Providers;
use rustc_middle::traits::ObligationCause;
use rustc_middle::ty::error::{ExpectedFound, TypeError};
use rustc_middle::ty::{self, TyCtxt, TypingMode, Unnormalized};
use rustc_middle::{bug, span_bug};
use rustc_session::config::CrateType;
use rustc_session::diagnostics::feature_err;
use rustc_session::lint;
use rustc_session::lint::builtin::{
    CONFLICTING_REPR_HINTS, INVALID_DOC_ATTRIBUTES, MALFORMED_DIAGNOSTIC_ATTRIBUTES,
    MALFORMED_DIAGNOSTIC_FORMAT_LITERALS, MISPLACED_DIAGNOSTIC_ATTRIBUTES, UNUSED_ATTRIBUTES,
};
use rustc_span::edition::Edition;
use rustc_span::{DUMMY_SP, Ident, Span, Symbol, sym};
use rustc_trait_selection::error_reporting::InferCtxtErrorExt;
use rustc_trait_selection::infer::{TyCtxtInferExt, ValuePairs};
use rustc_trait_selection::traits::ObligationCtxt;

use crate::diagnostics;

#[derive(Diagnostic)]
#[diag("`#[diagnostic::on_const]` can only be applied to non-const trait implementations")]
struct DiagnosticOnConstOnlyForNonConstTraitImpls {
    #[label("this is a const trait implementation")]
    item_span: Span,
}

fn target_from_impl_item<'tcx>(tcx: TyCtxt<'tcx>, impl_item: &hir::ImplItem<'_>) -> Target {
    match impl_item.kind {
        hir::ImplItemKind::Const(..) => Target::AssocConst,
        hir::ImplItemKind::Fn(..) => {
            let parent_def_id = tcx.hir_get_parent_item(impl_item.hir_id()).def_id;
            let containing_item = tcx.hir_expect_item(parent_def_id);
            let containing_impl_is_for_trait = match &containing_item.kind {
                hir::ItemKind::Impl(impl_) => impl_.of_trait.is_some(),
                _ => bug!("parent of an ImplItem must be an Impl"),
            };
            if containing_impl_is_for_trait {
                Target::Method(MethodKind::Trait { body: true })
            } else {
                Target::Method(MethodKind::Inherent)
            }
        }
        hir::ImplItemKind::Type(..) => Target::AssocTy,
    }
}

#[derive(Copy, Clone)]
pub(crate) enum ProcMacroKind {
    FunctionLike,
    Derive,
    Attribute,
}

impl IntoDiagArg for ProcMacroKind {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> rustc_errors::DiagArgValue {
        match self {
            ProcMacroKind::Attribute => "attribute proc macro",
            ProcMacroKind::Derive => "derive proc macro",
            ProcMacroKind::FunctionLike => "function-like proc macro",
        }
        .into_diag_arg(&mut None)
    }
}

struct CheckAttrVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,

    // Whether or not this visitor should abort after finding errors
    abort: Cell<bool>,
}

impl<'tcx> CheckAttrVisitor<'tcx> {
    fn dcx(&self) -> DiagCtxtHandle<'tcx> {
        self.tcx.dcx()
    }

    /// Checks any attribute.
    fn check_attributes(
        &self,
        hir_id: HirId,
        span: Span,
        target: Target,
        item: Option<&'tcx Item<'tcx>>,
    ) {
        let attrs = self.tcx.hir_attrs(hir_id);
        for attr in attrs {
            match attr {
                Attribute::Parsed(attr_kind) => {
                    self.check_one_parsed_attribute(hir_id, span, target, item, attrs, attr_kind);
                    self.check_unused_attribute(hir_id, attr, None);
                }
                Attribute::Unparsed(attr_item) => {
                    match attr.path().as_slice() {
                        // ok
                        [sym::allow | sym::expect | sym::warn | sym::deny | sym::forbid, ..] => {}

                        [name, rest @ ..] => {
                            if let Some(_) = BUILTIN_ATTRIBUTE_MAP.get(name) {
                                if rest.len() > 0
                                    && AttributeParser::is_parsed_attribute(slice::from_ref(name))
                                {
                                    // Check if we tried to use a builtin attribute as an attribute
                                    // namespace, like `#[must_use::skip]`. This check is here to
                                    // solve <https://github.com/rust-lang/rust/issues/137590>.
                                    // An error is already produced for this case elsewhere.
                                    return;
                                }

                                span_bug!(
                                    attr.span(),
                                    "builtin attribute {name:?} not handled by `CheckAttrVisitor`"
                                )
                            }
                        }

                        [] => unreachable!(),
                    }

                    self.check_unused_attribute(hir_id, attr, Some(attr_item.style));
                }
            }
        }

        self.check_repr(attrs, span, target, item, hir_id);
        self.check_rustc_force_inline(hir_id, attrs, target);
        self.check_mix_no_mangle_export(hir_id, attrs);
        self.check_optimize_and_inline(attrs);
    }

    /// Called by [`Self::check_attributes()`] to check a single attribute which is
    /// [`Attribute::Parsed`].
    ///
    /// This is a separate function to help with comprehensibility and rustfmt-ability.
    fn check_one_parsed_attribute(
        &self,
        hir_id: HirId,
        span: Span,
        target: Target,
        item: Option<&'tcx Item<'tcx>>,
        attrs: &[Attribute],
        attr: &AttributeKind,
    ) {
        match attr {
            AttributeKind::ProcMacro => {
                self.check_proc_macro(hir_id, target, ProcMacroKind::FunctionLike)
            }
            AttributeKind::ProcMacroAttribute => {
                self.check_proc_macro(hir_id, target, ProcMacroKind::Attribute);
            }
            AttributeKind::ProcMacroDerive { .. } => {
                self.check_proc_macro(hir_id, target, ProcMacroKind::Derive)
            }
            AttributeKind::Inline(InlineAttr::Force { .. }, ..) => {} // handled separately below
            AttributeKind::Inline(kind, attr_span) => {
                self.check_inline(hir_id, *attr_span, kind, target)
            }
            AttributeKind::AllowInternalUnsafe(attr_span)
            | AttributeKind::AllowInternalUnstable(.., attr_span) => {
                self.check_macro_only_attr(*attr_span, span, target, attrs)
            }
            AttributeKind::RustcAllowConstFnUnstable(_, first_span) => {
                self.check_rustc_allow_const_fn_unstable(hir_id, *first_span, span, target)
            }
            AttributeKind::Deprecated { span: attr_span, .. } => {
                self.check_deprecated(hir_id, *attr_span, target)
            }
            AttributeKind::RustcDumpObjectLifetimeDefaults => {
                self.check_dump_object_lifetime_defaults(hir_id);
            }
            AttributeKind::Naked(..) => self.check_naked(hir_id, target),
            AttributeKind::TrackCaller(attr_span) => {
                self.check_track_caller(hir_id, *attr_span, attrs, target)
            }
            AttributeKind::NonExhaustive(attr_span) => {
                self.check_non_exhaustive(*attr_span, span, target, item)
            }
            AttributeKind::MayDangle(attr_span) => self.check_may_dangle(hir_id, *attr_span),
            AttributeKind::Link(_, attr_span) => self.check_link(hir_id, *attr_span, target),
            AttributeKind::MacroExport { span, .. } => {
                self.check_macro_export(hir_id, *span, target)
            }
            AttributeKind::RustcLegacyConstGenerics { attr_span, fn_indexes } => {
                self.check_rustc_legacy_const_generics(item, *attr_span, fn_indexes)
            }
            AttributeKind::Doc(attr) => self.check_doc_attrs(attr, hir_id, target),
            AttributeKind::EiiImpls(impls) => self.check_eii_impl(impls, target),
            AttributeKind::RustcMustImplementOneOf { attr_span, fn_names } => {
                self.check_rustc_must_implement_one_of(*attr_span, fn_names, hir_id, target)
            }
            AttributeKind::OnUnimplemented { directive } => {
                self.check_diagnostic_on_unimplemented(hir_id, directive.as_deref())
            }
            AttributeKind::OnConst { span, directive } => {
                self.check_diagnostic_on_const(*span, hir_id, target, item, directive.as_deref())
            }
            AttributeKind::OnMove { directive } => {
                self.check_diagnostic_on_move(hir_id, directive.as_deref())
            }
            AttributeKind::OnTypeError { directive, .. } => {
                self.check_diagnostic_on_type_error(hir_id, directive.as_deref())
            }

            // All of the following attributes have no specific checks.
            // tidy-alphabetical-start
            AttributeKind::AutomaticallyDerived => (),
            AttributeKind::CfgAttrTrace => (),
            AttributeKind::CfgTrace(..) => (),
            AttributeKind::CfiEncoding { .. } => (),
            AttributeKind::Cold => (),
            AttributeKind::CollapseDebugInfo(..) => (),
            AttributeKind::CompilerBuiltins => (),
            AttributeKind::ConstContinue(..) => {}
            AttributeKind::Coroutine => (),
            AttributeKind::Coverage(..) => (),
            AttributeKind::CrateName { .. } => (),
            AttributeKind::CrateType(..) => (),
            AttributeKind::CustomMir(..) => (),
            AttributeKind::DebuggerVisualizer(..) => (),
            AttributeKind::DefaultLibAllocator => (),
            AttributeKind::DoNotRecommend => (),
            // `#[doc]` is actually a lot more than just doc comments, so is checked below
            AttributeKind::DocComment { .. } => (),
            AttributeKind::EiiDeclaration { .. } => (),
            AttributeKind::ExportName { .. } => (),
            AttributeKind::ExportStable => (),
            AttributeKind::Feature(..) => (),
            AttributeKind::FfiConst => (),
            AttributeKind::FfiPure(..) => (),
            AttributeKind::Fundamental => (),
            AttributeKind::Ignore { .. } => (),
            AttributeKind::InstructionSet(..) => (),
            AttributeKind::InstrumentFn(..) => (),
            AttributeKind::Lang(..) => (),
            AttributeKind::LinkName { .. } => (),
            AttributeKind::LinkOrdinal { .. } => (),
            AttributeKind::LinkSection { .. } => (),
            AttributeKind::Linkage(..) => (),
            AttributeKind::LoopMatch(..) => {}
            AttributeKind::MacroEscape => (),
            AttributeKind::MacroUse { .. } => (),
            AttributeKind::Marker => (),
            AttributeKind::MoveSizeLimit { .. } => (),
            AttributeKind::MustNotSupend { .. } => (),
            AttributeKind::MustUse { .. } => (),
            AttributeKind::NeedsAllocator => (),
            AttributeKind::NeedsPanicRuntime => (),
            AttributeKind::NoBuiltins => (),
            AttributeKind::NoCore { .. } => (),
            AttributeKind::NoImplicitPrelude => (),
            AttributeKind::NoLink => (),
            AttributeKind::NoMain => (),
            AttributeKind::NoMangle(..) => (),
            AttributeKind::NoStd { .. } => (),
            AttributeKind::OnUnknown { .. } => (),
            AttributeKind::OnUnmatchedArgs { .. } => (),
            AttributeKind::Opaque => (),
            AttributeKind::Optimize(..) => (),
            AttributeKind::PanicRuntime => (),
            AttributeKind::PatchableFunctionEntry { .. } => (),
            AttributeKind::Path(_, span) => self.check_path(*span, hir_id),
            AttributeKind::PatternComplexityLimit { .. } => (),
            AttributeKind::PinV2(..) => (),
            AttributeKind::PreludeImport => (),
            AttributeKind::ProfilerRuntime => (),
            AttributeKind::RecursionLimit { .. } => (),
            AttributeKind::ReexportTestHarnessMain(..) => (),
            AttributeKind::RegisterTool(..) => (),
            // handled below this loop and elsewhere
            AttributeKind::Repr { .. } => (),
            AttributeKind::RustcAbi { .. } => (),
            AttributeKind::RustcAlign { .. } => {}
            AttributeKind::RustcAllocator => (),
            AttributeKind::RustcAllocatorZeroed => (),
            AttributeKind::RustcAllocatorZeroedVariant { .. } => (),
            AttributeKind::RustcAllowIncoherentImpl(..) => (),
            AttributeKind::RustcAsPtr => (),
            AttributeKind::RustcAutodiff(..) => (),
            AttributeKind::RustcBodyStability { .. } => (),
            AttributeKind::RustcBuiltinMacro { .. } => (),
            AttributeKind::RustcCanonicalSymbol => (),
            AttributeKind::RustcCaptureAnalysis => (),
            AttributeKind::RustcCguTestAttr(..) => (),
            AttributeKind::RustcClean(..) => (),
            AttributeKind::RustcCoherenceIsCore => (),
            AttributeKind::RustcCoinductive => (),
            AttributeKind::RustcComptime(_) => (),
            AttributeKind::RustcConfusables { .. } => (),
            AttributeKind::RustcConstStability { .. } => (),
            AttributeKind::RustcConstStableIndirect => (),
            AttributeKind::RustcConversionSuggestion => (),
            AttributeKind::RustcDeallocator => (),
            AttributeKind::RustcDelayedBugFromInsideQuery => (),
            AttributeKind::RustcDenyExplicitImpl => (),
            AttributeKind::RustcDeprecatedSafe2024 { .. } => (),
            AttributeKind::RustcDiagnosticItem(..) => (),
            AttributeKind::RustcDoNotConstCheck => (),
            AttributeKind::RustcDocPrimitive(..) => (),
            AttributeKind::RustcDummy => (),
            AttributeKind::RustcDumpDefParents => (),
            AttributeKind::RustcDumpDefPath(..) => (),
            AttributeKind::RustcDumpGenerics => (),
            AttributeKind::RustcDumpHiddenTypeOfOpaques => (),
            AttributeKind::RustcDumpInferredOutlives => (),
            AttributeKind::RustcDumpItemBounds => (),
            AttributeKind::RustcDumpLayout(..) => (),
            AttributeKind::RustcDumpPredicates => (),
            AttributeKind::RustcDumpSymbolName(..) => (),
            AttributeKind::RustcDumpUserArgs => (),
            AttributeKind::RustcDumpVariances => (),
            AttributeKind::RustcDumpVariancesOfOpaques => (),
            AttributeKind::RustcDumpVtable(..) => (),
            AttributeKind::RustcDynIncompatibleTrait(..) => (),
            AttributeKind::RustcEffectiveVisibility => (),
            AttributeKind::RustcEiiForeignItem => (),
            AttributeKind::RustcEvaluateWhereClauses => (),
            AttributeKind::RustcHasIncoherentInherentImpls => (),
            AttributeKind::RustcIfThisChanged(..) => (),
            AttributeKind::RustcInheritOverflowChecks => (),
            AttributeKind::RustcInsignificantDtor => (),
            AttributeKind::RustcIntrinsic => (),
            AttributeKind::RustcIntrinsicConstStableIndirect => (),
            AttributeKind::RustcLintOptDenyFieldAccess { .. } => (),
            AttributeKind::RustcLintOptTy => (),
            AttributeKind::RustcLintQueryInstability => (),
            AttributeKind::RustcLintUntrackedQueryInformation => (),
            AttributeKind::RustcMacroTransparency(_) => (),
            AttributeKind::RustcMain => (),
            AttributeKind::RustcMir(_) => (),
            AttributeKind::RustcMustMatchExhaustively(..) => (),
            AttributeKind::RustcNeverReturnsNullPtr => (),
            AttributeKind::RustcNeverTypeOptions { .. } => (),
            AttributeKind::RustcNoImplicitAutorefs => (),
            AttributeKind::RustcNoImplicitBounds => (),
            AttributeKind::RustcNoMirInline => (),
            AttributeKind::RustcNoWritable => (),
            AttributeKind::RustcNonConstTraitMethod => (),
            AttributeKind::RustcNonnullOptimizationGuaranteed => (),
            AttributeKind::RustcNounwind => (),
            AttributeKind::RustcObjcClass { .. } => (),
            AttributeKind::RustcObjcSelector { .. } => (),
            AttributeKind::RustcOffloadKernel => (),
            AttributeKind::RustcParenSugar => (),
            AttributeKind::RustcPassByValue => (),
            AttributeKind::RustcPassIndirectlyInNonRusticAbis(..) => (),
            AttributeKind::RustcPreserveUbChecks => (),
            AttributeKind::RustcProcMacroDecls => (),
            AttributeKind::RustcPubTransparent(..) => (),
            AttributeKind::RustcReallocator => (),
            AttributeKind::RustcRegions => (),
            AttributeKind::RustcReservationImpl(..) => (),
            AttributeKind::RustcScalableVector { .. } => (),
            AttributeKind::RustcShouldNotBeCalledOnConstItems => (),
            AttributeKind::RustcSimdMonomorphizeLaneLimit(..) => (),
            AttributeKind::RustcSkipDuringMethodDispatch { .. } => (),
            AttributeKind::RustcSpecializationTrait => (),
            AttributeKind::RustcStdInternalSymbol => (),
            AttributeKind::RustcStrictCoherence(..) => (),
            AttributeKind::RustcTestEntrypointMarker => (),
            AttributeKind::RustcTestMarker(..) => (),
            AttributeKind::RustcThenThisWouldNeed(..) => (),
            AttributeKind::RustcTrivialFieldReads => (),
            AttributeKind::RustcUnsafeSpecializationMarker => (),
            AttributeKind::Sanitize { .. } => {}
            AttributeKind::ShouldPanic { .. } => (),
            AttributeKind::Splat(..) => (),
            AttributeKind::Stability { .. } => (),
            AttributeKind::TargetFeature { .. } => {}
            AttributeKind::TestRunner(..) => (),
            AttributeKind::ThreadLocal => (),
            AttributeKind::TypeLengthLimit { .. } => (),
            AttributeKind::Unroll(..) => (),
            AttributeKind::UnstableFeatureBound(..) => (),
            AttributeKind::UnstableRemoved(..) => (),
            AttributeKind::Used { .. } => (),
            AttributeKind::WindowsSubsystem(..) => (),
            // tidy-alphabetical-end
        }
    }

    fn check_path(&self, span: Span, hir_id: HirId) {
        let Node::Item(item) = self.tcx.hir_node(hir_id) else {
            return;
        };

        let ItemKind::Mod(_, module) = &item.kind else {
            return;
        };

        if !item.span.contains(module.spans.inner_span) {
            return;
        }

        let has_out_of_line_child_module = module.item_ids.iter().any(|item_id| {
            let child = self.tcx.hir_item(*item_id);

            let ItemKind::Mod(_, child_mod) = &child.kind else {
                return false;
            };

            !child.span.contains(child_mod.spans.inner_span)
        });

        if has_out_of_line_child_module {
            return;
        }

        let has_child_module_with_path_attr = module.item_ids.iter().any(|item_id| {
            let child = self.tcx.hir_item(*item_id);

            matches!(child.kind, ItemKind::Mod(..))
                && find_attr!(self.tcx, child.hir_id(), Path(..))
        });

        if has_child_module_with_path_attr {
            return;
        }

        self.tcx.emit_node_span_lint(
            UNUSED_ATTRIBUTES,
            hir_id,
            span,
            diagnostics::Unused {
                attr_span: span,
                note: diagnostics::UnusedNote::PathOnInlineModule,
            },
        );
    }

    fn check_rustc_must_implement_one_of(
        &self,
        attr_span: Span,
        list: &ThinVec<Ident>,
        hir_id: HirId,
        target: Target,
    ) {
        // Ignoring invalid targets because TyCtxt::associated_items emits bug if the target isn't valid
        // the parser has already produced an error for the target being invalid
        if !matches!(target, Target::Trait) {
            return;
        }

        let def_id = hir_id.owner.def_id;

        let items = self.tcx.associated_items(def_id);
        // Check that all arguments of `#[rustc_must_implement_one_of]` reference
        // functions in the trait with default implementations
        for ident in list {
            let item = items
                .filter_by_name_unhygienic(ident.name)
                .find(|item| item.ident(self.tcx) == *ident);

            match item {
                Some(item) if matches!(item.kind, ty::AssocKind::Fn { .. }) => {
                    if !item.defaultness(self.tcx).has_value() {
                        self.tcx.dcx().emit_err(
                            diagnostics::FunctionNotHaveDefaultImplementation {
                                span: self.tcx.def_span(item.def_id),
                                note_span: attr_span,
                            },
                        );
                    }
                }
                Some(item) => {
                    self.dcx().emit_err(diagnostics::MustImplementNotFunction {
                        span: self.tcx.def_span(item.def_id),
                        span_note: diagnostics::MustImplementNotFunctionSpanNote {
                            span: attr_span,
                        },
                        note: diagnostics::MustImplementNotFunctionNote {},
                    });
                }
                None => {
                    self.dcx().emit_err(diagnostics::FunctionNotFoundInTrait { span: ident.span });
                }
            }
        }
        // Check for duplicates

        let mut set: UnordMap<Symbol, Span> = Default::default();

        for ident in &*list {
            if let Some(dup) = set.insert(ident.name, ident.span) {
                self.tcx.dcx().emit_err(diagnostics::FunctionNamesDuplicated {
                    spans: vec![dup, ident.span],
                });
            }
        }
    }

    fn check_eii_impl(&self, impls: &[EiiImpl], target: Target) {
        for EiiImpl { span, inner_span, resolution, impl_marked_unsafe, is_default: _ } in impls {
            match target {
                Target::Fn | Target::Static => {}
                _ => {
                    self.dcx().emit_err(diagnostics::EiiImplTarget { span: *span });
                }
            }

            let needs_unsafe = match resolution {
                EiiImplResolution::Macro(eii_macro) => {
                    find_attr!(self.tcx, *eii_macro, EiiDeclaration(EiiDecl { impl_unsafe, .. }) if *impl_unsafe)
                }
                EiiImplResolution::Known(foreign_item_did) => {
                    let foreign_item_did = *foreign_item_did;
                    self.tcx
                        .externally_implementable_items(foreign_item_did.krate)
                        .get(&foreign_item_did)
                        .map(|(decl, _)| decl.impl_unsafe)
                        .unwrap_or(false)
                }
                EiiImplResolution::Error(_) => false,
            };

            if needs_unsafe && !impl_marked_unsafe {
                let name = match resolution {
                    EiiImplResolution::Macro(eii_macro) => self.tcx.item_name(*eii_macro),
                    EiiImplResolution::Known(def_id) => self.tcx.item_name(*def_id),
                    EiiImplResolution::Error(_) => unreachable!(),
                };
                self.dcx().emit_err(diagnostics::EiiImplRequiresUnsafe {
                    span: *span,
                    name,
                    suggestion: diagnostics::EiiImplRequiresUnsafeSuggestion {
                        left: inner_span.shrink_to_lo(),
                        right: inner_span.shrink_to_hi(),
                    },
                });
            }
        }
    }

    /// Checks use of generic formatting parameters in `#[diagnostic::on_unimplemented]`
    fn check_diagnostic_on_unimplemented(&self, hir_id: HirId, directive: Option<&Directive>) {
        if let Some(directive) = directive {
            if let Node::Item(Item {
                kind: ItemKind::Trait { ident: trait_name, generics, .. },
                ..
            }) = self.tcx.hir_node(hir_id)
            {
                directive.visit_params(&mut |argument_name, span| {
                    let has_generic = generics.params.iter().any(|p| {
                        if !matches!(p.kind, GenericParamKind::Lifetime { .. })
                            && let ParamName::Plain(name) = p.name
                            && name.name == argument_name
                        {
                            true
                        } else {
                            false
                        }
                    });
                    if !has_generic {
                        self.tcx.emit_node_span_lint(
                            MALFORMED_DIAGNOSTIC_FORMAT_LITERALS,
                            hir_id,
                            span,
                            diagnostics::UnknownFormatParameterForOnUnimplementedAttr {
                                argument_name,
                                trait_name: *trait_name,
                                help: !directive.is_rustc_attr,
                            },
                        )
                    }
                })
            }
        }
    }

    /// Checks if `#[diagnostic::on_const]` is applied to a on-const trait impl
    fn check_diagnostic_on_const(
        &self,
        attr_span: Span,
        hir_id: HirId,
        target: Target,
        item: Option<&'tcx Item<'tcx>>,
        directive: Option<&Directive>,
    ) {
        // We only check the non-constness here. A diagnostic for use
        // on not-trait impl items is issued during attribute parsing.
        if target == (Target::Impl { of_trait: true }) {
            if let Some(directive) = directive
                && let Node::Item(Item { kind: ItemKind::Impl(hir::Impl { generics, .. }), .. }) =
                    self.tcx.hir_node(hir_id)
            {
                directive.visit_params(&mut |argument_name, span| {
                    let has_generic = generics.params.iter().any(|p| {
                        if !matches!(p.kind, GenericParamKind::Lifetime { .. })
                            && let ParamName::Plain(name) = p.name
                            && name.name == argument_name
                        {
                            true
                        } else {
                            false
                        }
                    });
                    if !has_generic {
                        self.tcx.emit_node_span_lint(
                            MALFORMED_DIAGNOSTIC_FORMAT_LITERALS,
                            hir_id,
                            span,
                            diagnostics::OnConstMalformedFormatLiterals { name: argument_name },
                        )
                    }
                });
            }
            match item.unwrap().expect_impl().constness {
                Constness::Const { .. } => {
                    let item_span = self.tcx.hir_span(hir_id);
                    self.tcx.emit_node_span_lint(
                        MISPLACED_DIAGNOSTIC_ATTRIBUTES,
                        hir_id,
                        attr_span,
                        DiagnosticOnConstOnlyForNonConstTraitImpls { item_span },
                    );
                    return;
                }
                Constness::NotConst => return,
            }
        }
    }

    /// Checks use of generic formatting parameters in `#[diagnostic::on_move]`
    fn check_diagnostic_on_move(&self, hir_id: HirId, directive: Option<&Directive>) {
        if let Some(directive) = directive {
            if let Node::Item(Item {
                kind:
                    ItemKind::Struct(_, generics, _)
                    | ItemKind::Enum(_, generics, _)
                    | ItemKind::Union(_, generics, _),
                ..
            }) = self.tcx.hir_node(hir_id)
            {
                directive.visit_params(&mut |argument_name, span| {
                    let has_generic = generics.params.iter().any(|p| {
                        if !matches!(p.kind, GenericParamKind::Lifetime { .. })
                            && let ParamName::Plain(name) = p.name
                            && name.name == argument_name
                        {
                            true
                        } else {
                            false
                        }
                    });
                    if !has_generic {
                        self.tcx.emit_node_span_lint(
                            MALFORMED_DIAGNOSTIC_FORMAT_LITERALS,
                            hir_id,
                            span,
                            diagnostics::OnMoveMalformedFormatLiterals { name: argument_name },
                        )
                    }
                });
            }
        }
    }

    fn check_diagnostic_on_type_error(&self, hir_id: HirId, directive: Option<&Directive>) {
        if let Some(directive) = directive {
            if let Node::Item(Item {
                kind:
                    ItemKind::Struct(_, generics, _)
                    | ItemKind::Enum(_, generics, _)
                    | ItemKind::Union(_, generics, _),
                ..
            }) = self.tcx.hir_node(hir_id)
            {
                let generic_count = generics
                    .params
                    .iter()
                    .filter(|p| !matches!(p.kind, GenericParamKind::Lifetime { .. }))
                    .count();

                // Enforce: at most one generic
                if generic_count != 1 {
                    self.tcx.emit_node_span_lint(
                        MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                        hir_id,
                        generics.span,
                        diagnostics::OnTypeErrorNotExactlyOneGeneric { count: generic_count },
                    );
                }

                directive.visit_params(&mut |argument_name, span| {
                    let has_generic = generics.params.iter().any(|p| {
                        if !matches!(p.kind, GenericParamKind::Lifetime { .. })
                            && let ParamName::Plain(name) = p.name
                            && name.name == argument_name
                        {
                            true
                        } else {
                            false
                        }
                    });

                    let is_allowed = argument_name == sym::Expected || argument_name == sym::Found;
                    if !(has_generic | is_allowed) {
                        self.tcx.emit_node_span_lint(
                            MALFORMED_DIAGNOSTIC_FORMAT_LITERALS,
                            hir_id,
                            span,
                            diagnostics::OnTypeErrorMalformedFormatLiterals { name: argument_name },
                        )
                    }
                });
            }
        }
    }

    /// Checks if an `#[inline]` is applied to a function or a closure.
    fn check_inline(&self, hir_id: HirId, attr_span: Span, kind: &InlineAttr, target: Target) {
        match target {
            Target::Fn
            | Target::Closure
            | Target::Method(MethodKind::Trait { body: true } | MethodKind::Inherent) => {
                // `#[inline]` is ignored if the symbol must be codegened upstream because it's exported.
                if let Some(did) = hir_id.as_owner()
                    && self.tcx.def_kind(did).has_codegen_attrs()
                    && kind != &InlineAttr::Never
                {
                    let attrs = self.tcx.codegen_fn_attrs(did);
                    // Not checking naked as `#[inline]` is forbidden for naked functions anyways.
                    if attrs.contains_extern_indicator() {
                        self.tcx.emit_node_span_lint(
                            UNUSED_ATTRIBUTES,
                            hir_id,
                            attr_span,
                            diagnostics::InlineIgnoredForExported,
                        );
                    }
                }
            }
            _ => {}
        }
    }

    /// Checks if `#[naked]` is applied to a function definition.
    fn check_naked(&self, hir_id: HirId, target: Target) {
        match target {
            Target::Fn
            | Target::Method(MethodKind::Trait { body: true } | MethodKind::Inherent) => {
                let fn_sig = self.tcx.hir_node(hir_id).fn_sig().unwrap();
                let abi = fn_sig.header.abi;
                if abi.is_rustic_abi() && !self.tcx.features().naked_functions_rustic_abi() {
                    feature_err(
                        &self.tcx.sess,
                        sym::naked_functions_rustic_abi,
                        fn_sig.span,
                        format!(
                            "`#[naked]` is currently unstable on `extern \"{}\"` functions",
                            abi.as_str()
                        ),
                    )
                    .emit();
                }
            }
            _ => {}
        }
    }

    /// Debugging aid for the `object_lifetime_default` query.
    fn check_dump_object_lifetime_defaults(&self, hir_id: HirId) {
        let tcx = self.tcx;
        let Some(owner_id) = hir_id.as_owner() else { return };
        for param in &tcx.generics_of(owner_id.def_id).own_params {
            let ty::GenericParamDefKind::Type { .. } = param.kind else { continue };
            let default = tcx.object_lifetime_default(param.def_id);
            let repr = match default {
                ObjectLifetimeDefault::Empty => "Empty".to_owned(),
                ObjectLifetimeDefault::Static => "'static".to_owned(),
                ObjectLifetimeDefault::Param(def_id) => tcx.item_name(def_id).to_string(),
                ObjectLifetimeDefault::Ambiguous => "Ambiguous".to_owned(),
            };
            tcx.dcx().span_err(tcx.def_span(param.def_id), repr);
        }
    }

    /// Checks if a `#[track_caller]` is applied to a function.
    fn check_track_caller(
        &self,
        hir_id: HirId,
        attr_span: Span,
        attrs: &[Attribute],
        target: Target,
    ) {
        match target {
            Target::Fn => {
                // `#[track_caller]` is not valid on weak lang items because they are called via
                // `extern` declarations and `#[track_caller]` would alter their ABI.
                if let Some(item) = find_attr!(attrs, Lang(item) => item)
                    && item.is_weak()
                {
                    let sig = self.tcx.hir_node(hir_id).fn_sig().unwrap();

                    self.dcx().emit_err(diagnostics::LangItemWithTrackCaller {
                        attr_span,
                        name: item.name(),
                        sig_span: sig.span,
                    });
                }

                if let Some(impls) = find_attr!(attrs, EiiImpls(impls) => impls) {
                    let sig = self.tcx.hir_node(hir_id).fn_sig().unwrap();
                    for i in impls {
                        let name = match i.resolution {
                            EiiImplResolution::Macro(def_id) => self.tcx.item_name(def_id),
                            EiiImplResolution::Known(def_id) => self.tcx.item_name(def_id),
                            EiiImplResolution::Error(_eg) => continue,
                        };
                        self.dcx().emit_err(diagnostics::EiiWithTrackCaller {
                            attr_span,
                            name,
                            sig_span: sig.span,
                        });
                    }
                }
            }
            _ => {}
        }
    }

    /// Checks if the `#[non_exhaustive]` attribute on an `item` is valid.
    fn check_non_exhaustive(
        &self,
        attr_span: Span,
        span: Span,
        target: Target,
        item: Option<&'tcx Item<'tcx>>,
    ) {
        match target {
            Target::Struct => {
                if let hir::Item {
                    kind: hir::ItemKind::Struct(_, _, hir::VariantData::Struct { fields, .. }),
                    ..
                } = item.unwrap()
                    && !fields.is_empty()
                    && fields.iter().any(|f| f.default.is_some())
                {
                    self.dcx().emit_err(diagnostics::NonExhaustiveWithDefaultFieldValues {
                        attr_span,
                        defn_span: span,
                    });
                }
            }
            _ => {}
        }
    }

    fn check_doc_alias_value(&self, span: Span, hir_id: HirId, target: Target, alias: Symbol) {
        if let Some(location) = match target {
            Target::AssocTy => {
                if let DefKind::Impl { .. } =
                    self.tcx.def_kind(self.tcx.local_parent(hir_id.owner.def_id))
                {
                    Some("type alias in implementation block")
                } else {
                    None
                }
            }
            Target::AssocConst => {
                let parent_def_id = self.tcx.hir_get_parent_item(hir_id).def_id;
                let containing_item = self.tcx.hir_expect_item(parent_def_id);
                // We can't link to trait impl's consts.
                let err = "associated constant in trait implementation block";
                match containing_item.kind {
                    ItemKind::Impl(hir::Impl { of_trait: Some(_), .. }) => Some(err),
                    _ => None,
                }
            }
            // we check the validity of params elsewhere
            Target::Param => return,
            Target::Expression
            | Target::Statement
            | Target::Arm
            | Target::ForeignMod
            | Target::Closure
            | Target::Impl { .. }
            | Target::WherePredicate => Some(target.name()),
            Target::ExternCrate
            | Target::Use
            | Target::Static
            | Target::Const
            | Target::Fn
            | Target::Mod
            | Target::GlobalAsm
            | Target::TyAlias
            | Target::Enum
            | Target::Variant
            | Target::Struct
            | Target::Field
            | Target::Union
            | Target::Trait
            | Target::TraitAlias
            | Target::Method(..)
            | Target::ForeignFn
            | Target::ForeignStatic
            | Target::ForeignTy
            | Target::GenericParam { .. }
            | Target::MacroDef
            | Target::PatField
            | Target::ExprField
            | Target::Crate
            | Target::MacroCall
            | Target::Delegation { .. }
            | Target::Loop
            | Target::ForLoop
            | Target::While
            | Target::Break => None,
        } {
            self.tcx.dcx().emit_err(diagnostics::DocAliasBadLocation { span, location });
            return;
        }
        if self.tcx.hir_opt_name(hir_id) == Some(alias) {
            self.tcx.dcx().emit_err(diagnostics::DocAliasNotAnAlias { span, attr_str: alias });
            return;
        }
    }

    fn check_doc_fake_variadic(&self, span: Span, hir_id: HirId) {
        let item_kind = match self.tcx.hir_node(hir_id) {
            hir::Node::Item(item) => Some(&item.kind),
            _ => None,
        };
        match item_kind {
            Some(ItemKind::Impl(i)) => {
                let is_valid = doc_fake_variadic_is_allowed_self_ty(i.self_ty)
                    || if let Some(&[hir::GenericArg::Type(ty)]) = i
                        .of_trait
                        .and_then(|of_trait| of_trait.trait_ref.path.segments.last())
                        .map(|last_segment| last_segment.args().args)
                    {
                        matches!(&ty.kind, hir::TyKind::Tup([_]))
                    } else {
                        false
                    };
                if !is_valid {
                    self.dcx().emit_err(diagnostics::DocFakeVariadicNotValid { span });
                }
            }
            _ => {
                self.dcx().emit_err(diagnostics::DocKeywordOnlyImpl { span });
            }
        }
    }

    fn check_doc_search_unbox(&self, span: Span, hir_id: HirId) {
        let hir::Node::Item(item) = self.tcx.hir_node(hir_id) else {
            self.dcx().emit_err(diagnostics::DocSearchUnboxInvalid { span });
            return;
        };
        match item.kind {
            ItemKind::Enum(_, generics, _) | ItemKind::Struct(_, generics, _)
                if generics.params.len() != 0 => {}
            ItemKind::Trait { generics, items, .. }
                if generics.params.len() != 0
                    || items.iter().any(|item| {
                        matches!(self.tcx.def_kind(item.owner_id), DefKind::AssocTy)
                    }) => {}
            ItemKind::TyAlias(_, generics, _) if generics.params.len() != 0 => {}
            _ => {
                self.dcx().emit_err(diagnostics::DocSearchUnboxInvalid { span });
            }
        }
    }

    /// Checks `#[doc(inline)]`/`#[doc(no_inline)]` attributes.
    ///
    /// A doc inlining attribute is invalid if it is applied to a non-`use` item, or
    /// if there are conflicting attributes for one item.
    ///
    /// `specified_inline` is used to keep track of whether we have
    /// already seen an inlining attribute for this item.
    /// If so, `specified_inline` holds the value and the span of
    /// the first `inline`/`no_inline` attribute.
    fn check_doc_inline(&self, hir_id: HirId, target: Target, inline: &[(DocInline, Span)]) {
        let span = match inline {
            [] => return,
            [(_, span)] => *span,
            [(inline, span), rest @ ..] => {
                for (inline2, span2) in rest {
                    if inline2 != inline {
                        let mut spans = MultiSpan::from_spans(vec![*span, *span2]);
                        spans.push_span_label(*span, msg!("this attribute..."));
                        spans.push_span_label(
                            *span2,
                            msg!("{\".\"}..conflicts with this attribute"),
                        );
                        self.dcx().emit_err(diagnostics::DocInlineConflict { spans });
                        return;
                    }
                }
                *span
            }
        };

        match target {
            Target::Use | Target::ExternCrate => {}
            _ => {
                self.tcx.emit_node_span_lint(
                    INVALID_DOC_ATTRIBUTES,
                    hir_id,
                    span,
                    diagnostics::DocInlineOnlyUse {
                        attr_span: span,
                        item_span: self.tcx.hir_span(hir_id),
                    },
                );
            }
        }
    }

    fn check_doc_masked(&self, span: Span, hir_id: HirId, target: Target) {
        if target != Target::ExternCrate {
            self.tcx.emit_node_span_lint(
                INVALID_DOC_ATTRIBUTES,
                hir_id,
                span,
                diagnostics::DocMaskedOnlyExternCrate {
                    attr_span: span,
                    item_span: self.tcx.hir_span(hir_id),
                },
            );
            return;
        }

        if self.tcx.extern_mod_stmt_cnum(hir_id.owner.def_id).is_none() {
            self.tcx.emit_node_span_lint(
                INVALID_DOC_ATTRIBUTES,
                hir_id,
                span,
                diagnostics::DocMaskedNotExternCrateSelf {
                    attr_span: span,
                    item_span: self.tcx.hir_span(hir_id),
                },
            );
        }
    }

    fn check_doc_keyword_and_attribute(&self, span: Span, hir_id: HirId, attr_name: &'static str) {
        let item_kind = match self.tcx.hir_node(hir_id) {
            hir::Node::Item(item) => Some(&item.kind),
            _ => None,
        };
        match item_kind {
            Some(ItemKind::Mod(_, module)) => {
                if !module.item_ids.is_empty() {
                    self.dcx()
                        .emit_err(diagnostics::DocKeywordAttributeEmptyMod { span, attr_name });
                    return;
                }
            }
            _ => {
                self.dcx().emit_err(diagnostics::DocKeywordAttributeNotMod { span, attr_name });
                return;
            }
        }
    }

    /// Runs various checks on `#[doc]` attributes.
    ///
    /// `specified_inline` should be initialized to `None` and kept for the scope
    /// of one item. Read the documentation of [`check_doc_inline`] for more information.
    ///
    /// [`check_doc_inline`]: Self::check_doc_inline
    fn check_doc_attrs(&self, attr: &DocAttribute, hir_id: HirId, target: Target) {
        let DocAttribute {
            first_span: _,
            aliases,
            // valid pretty much anywhere, not checked here?
            // FIXME: should we?
            hidden: _,
            inline,
            // FIXME: currently unchecked
            cfg: _,
            // already checked in attr_parsing
            auto_cfg: _,
            // already checked in attr_parsing
            auto_cfg_change: _,
            fake_variadic,
            keyword,
            masked,
            // FIXME: currently unchecked
            notable_trait: _,
            search_unbox,
            // already checked in attr_parsing
            html_favicon_url: _,
            // already checked in attr_parsing
            html_logo_url: _,
            // already checked in attr_parsing
            html_playground_url: _,
            // already checked in attr_parsing
            html_root_url: _,
            // already checked in attr_parsing
            html_no_source: _,
            // already checked in attr_parsing
            issue_tracker_base_url: _,
            // already checked in attr_parsing
            rust_logo: _,
            // allowed anywhere
            test_attrs: _,
            // already checked in attr_parsing
            no_crate_inject: _,
            attribute,
        } = attr;

        for (alias, span) in aliases {
            self.check_doc_alias_value(*span, hir_id, target, *alias);
        }

        if let Some((_, span)) = keyword {
            self.check_doc_keyword_and_attribute(*span, hir_id, "keyword");
        }
        if let Some((_, span)) = attribute {
            self.check_doc_keyword_and_attribute(*span, hir_id, "attribute");
        }

        if let Some(span) = fake_variadic {
            self.check_doc_fake_variadic(*span, hir_id);
        }

        if let Some(span) = search_unbox {
            self.check_doc_search_unbox(*span, hir_id);
        }

        self.check_doc_inline(hir_id, target, inline);

        if let Some(span) = masked {
            self.check_doc_masked(*span, hir_id, target);
        }
    }

    /// Checks if `#[may_dangle]` is applied to a lifetime or type generic parameter in `Drop` impl.
    fn check_may_dangle(&self, hir_id: HirId, attr_span: Span) {
        let hir::Node::GenericParam(
            param @ GenericParam {
                kind: hir::GenericParamKind::Lifetime { .. } | hir::GenericParamKind::Type { .. },
                ..
            },
        ) = self.tcx.hir_node(hir_id)
        else {
            self.dcx().delayed_bug("Checked in attr parser");
            return;
        };

        if matches!(param.source, hir::GenericParamSource::Generics)
            && let parent_hir_id = self.tcx.parent_hir_id(hir_id)
            && let hir::Node::Item(item) = self.tcx.hir_node(parent_hir_id)
            && let hir::ItemKind::Impl(impl_) = item.kind
            && let Some(of_trait) = impl_.of_trait
            && let Some(def_id) = of_trait.trait_ref.trait_def_id()
            && self.tcx.is_lang_item(def_id, hir::LangItem::Drop)
        {
            return;
        }

        self.dcx().emit_err(diagnostics::InvalidMayDangle { attr_span });
    }

    /// Checks if `#[link]` is applied to an item other than a foreign module.
    fn check_link(&self, hir_id: HirId, attr_span: Span, target: Target) {
        if target != Target::ForeignMod {
            return; // Checked by attribute parser
        }

        if let hir::Node::Item(item) = self.tcx.hir_node(hir_id)
            && let Item { kind: ItemKind::ForeignMod { abi, .. }, .. } = item
            && !matches!(abi, ExternAbi::Rust)
        {
            return;
        }

        self.tcx.emit_node_span_lint(UNUSED_ATTRIBUTES, hir_id, attr_span, diagnostics::Link);
    }

    /// Checks if `#[rustc_legacy_const_generics]` is applied to a function and has a valid argument.
    fn check_rustc_legacy_const_generics(
        &self,
        item: Option<&'tcx Item<'tcx>>,
        attr_span: Span,
        index_list: &ThinVec<(usize, Span)>,
    ) {
        let Some(Item { kind: ItemKind::Fn { sig: FnSig { decl, .. }, generics, .. }, .. }) = item
        else {
            // No error here, since it's already given by the parser
            return;
        };

        for param in generics.params {
            match param.kind {
                hir::GenericParamKind::Const { .. } => {}
                _ => {
                    self.dcx().emit_err(diagnostics::RustcLegacyConstGenericsOnly {
                        attr_span,
                        param_span: param.span,
                    });
                    return;
                }
            }
        }

        if index_list.len() != generics.params.len() {
            self.dcx().emit_err(diagnostics::RustcLegacyConstGenericsIndex {
                attr_span,
                generics_span: generics.span,
            });
            return;
        }

        let arg_count = decl.inputs.len() + generics.params.len();
        for (index, span) in index_list {
            if *index >= arg_count {
                self.dcx().emit_err(diagnostics::RustcLegacyConstGenericsIndexExceed {
                    span: *span,
                    arg_count,
                });
            }
        }
    }

    /// Checks if the `#[repr]` attributes on `item` are valid.
    fn check_repr(
        &self,
        attrs: &[Attribute],
        span: Span,
        target: Target,
        item: Option<&'tcx Item<'tcx>>,
        hir_id: HirId,
    ) {
        // Extract the names of all repr hints, e.g., [foo, bar, align] for:
        // ```
        // #[repr(foo)]
        // #[repr(bar, align(8))]
        // ```
        let (reprs, _first_attr_span) =
            find_attr!(attrs, Repr { reprs, first_span } => (reprs.as_slice(), Some(*first_span)))
                .unwrap_or((&[], None));

        let mut int_reprs = 0;
        let mut is_explicit_rust = false;
        let mut is_c = false;
        let mut is_simd = false;
        let mut is_transparent = false;

        for (repr, _repr_span) in reprs {
            match repr {
                ReprAttr::ReprRust => {
                    is_explicit_rust = true;
                }
                ReprAttr::ReprC => {
                    is_c = true;
                }
                ReprAttr::ReprAlign(..) => {}
                ReprAttr::ReprPacked(_) => {}
                ReprAttr::ReprSimd => {
                    is_simd = true;
                }
                ReprAttr::ReprTransparent => {
                    is_transparent = true;
                }
                ReprAttr::ReprInt(_) => {
                    int_reprs += 1;
                }
            };
        }

        // Just point at all repr hints if there are any incompatibilities.
        // This is not ideal, but tracking precisely which ones are at fault is a huge hassle.
        let hint_spans = reprs.iter().map(|(_, span)| *span);

        // Error on repr(transparent, <anything else>).
        if is_transparent && reprs.len() > 1 {
            let hint_spans = hint_spans.clone().collect();
            self.dcx().emit_err(diagnostics::TransparentIncompatible {
                hint_spans,
                target: target.to_string(),
            });
        }
        // Error on `#[repr(transparent)]` in combination with
        // `#[rustc_pass_indirectly_in_non_rustic_abis]`
        if is_transparent
            && let Some(&pass_indirectly_span) =
                find_attr!(attrs, RustcPassIndirectlyInNonRusticAbis(span) => span)
        {
            self.dcx().emit_err(diagnostics::TransparentIncompatible {
                hint_spans: vec![span, pass_indirectly_span],
                target: target.to_string(),
            });
        }
        if is_explicit_rust && (int_reprs > 0 || is_c || is_simd) {
            let hint_spans = hint_spans.clone().collect();
            self.dcx().emit_err(diagnostics::ReprConflicting { hint_spans });
        }
        // Warn on repr(u8, u16), repr(C, simd), and c-like-enum-repr(C, u8)
        if (int_reprs > 1)
            || (is_simd && is_c)
            || (int_reprs == 1 && is_c && item.is_some_and(is_c_like_enum))
        {
            self.tcx.emit_node_span_lint(
                CONFLICTING_REPR_HINTS,
                hir_id,
                hint_spans.collect::<Vec<Span>>(),
                diagnostics::ReprConflictingLint,
            );
        }
    }

    /// Outputs an error for attributes that can only be applied to macros, such as
    /// `#[allow_internal_unsafe]` and `#[allow_internal_unstable]`.
    /// (Allows proc_macro functions)
    // FIXME(jdonszelmann): if possible, move to attr parsing
    fn check_macro_only_attr(
        &self,
        attr_span: Span,
        span: Span,
        target: Target,
        attrs: &[Attribute],
    ) {
        match target {
            Target::Fn => {
                for attr in attrs {
                    if attr.is_proc_macro_attr() {
                        // return on proc macros
                        return;
                    }
                }
                self.tcx.dcx().emit_err(diagnostics::MacroOnlyAttribute { attr_span, span });
            }
            _ => {}
        }
    }

    /// Outputs an error for `#[allow_internal_unstable]` which can only be applied to macros.
    /// (Allows proc_macro functions)
    fn check_rustc_allow_const_fn_unstable(
        &self,
        hir_id: HirId,
        attr_span: Span,
        span: Span,
        target: Target,
    ) {
        match target {
            Target::Fn | Target::Method(_) => {
                if !self.tcx.is_const_fn(hir_id.expect_owner().to_def_id()) {
                    self.tcx
                        .dcx()
                        .emit_err(diagnostics::RustcAllowConstFnUnstable { attr_span, span });
                }
            }
            _ => {}
        }
    }

    fn check_deprecated(&self, hir_id: HirId, attr_span: Span, target: Target) {
        match target {
            Target::AssocConst | Target::Method(..) | Target::AssocTy
                if self.tcx.def_kind(self.tcx.local_parent(hir_id.owner.def_id))
                    == DefKind::Impl { of_trait: true } =>
            {
                self.tcx.emit_node_span_lint(
                    UNUSED_ATTRIBUTES,
                    hir_id,
                    attr_span,
                    diagnostics::DeprecatedAnnotationHasNoEffect { span: attr_span },
                );
            }
            _ => {}
        }
    }

    fn check_macro_export(&self, hir_id: HirId, attr_span: Span, target: Target) {
        if target != Target::MacroDef {
            return;
        }

        // special case when `#[macro_export]` is applied to a macro 2.0
        let (_, macro_definition, _) = self.tcx.hir_node(hir_id).expect_item().expect_macro();
        let is_decl_macro = !macro_definition.macro_rules;

        if is_decl_macro {
            self.tcx.emit_node_span_lint(
                UNUSED_ATTRIBUTES,
                hir_id,
                attr_span,
                diagnostics::MacroExport::OnDeclMacro,
            );
        }
    }

    fn check_unused_attribute(&self, hir_id: HirId, attr: &Attribute, style: Option<AttrStyle>) {
        // Warn on useless empty attributes.
        // FIXME(jdonszelmann): this lint should be moved to attribute parsing, see `AcceptContext::warn_empty_attribute`
        let note =
            if attr.has_any_name(&[sym::allow, sym::expect, sym::warn, sym::deny, sym::forbid])
                && attr.meta_item_list().is_some_and(|list| list.is_empty())
            {
                diagnostics::UnusedNote::EmptyList { name: attr.name().unwrap() }
            } else if attr.has_any_name(&[
                sym::allow,
                sym::warn,
                sym::deny,
                sym::forbid,
                sym::expect,
            ]) && let Some(meta) = attr.meta_item_list()
                && let [meta] = meta.as_slice()
                && let Some(item) = meta.meta_item()
                && let MetaItemKind::NameValue(_) = &item.kind
                && item.path == sym::reason
            {
                diagnostics::UnusedNote::NoLints { name: attr.name().unwrap() }
            } else if attr.has_any_name(&[
                sym::allow,
                sym::warn,
                sym::deny,
                sym::forbid,
                sym::expect,
            ]) && let Some(meta) = attr.meta_item_list()
                && meta.iter().any(|meta| {
                    meta.meta_item().map_or(false, |item| {
                        item.path == sym::linker_messages || item.path == sym::linker_info
                    })
                })
            {
                if hir_id != CRATE_HIR_ID {
                    match style {
                        Some(ast::AttrStyle::Outer) => {
                            let attr_span = attr.span();
                            let bang_position = self
                                .tcx
                                .sess
                                .source_map()
                                .span_until_char(attr_span, '[')
                                .shrink_to_hi();

                            self.tcx.emit_node_span_lint(
                                UNUSED_ATTRIBUTES,
                                hir_id,
                                attr_span,
                                diagnostics::OuterCrateLevelAttr {
                                    suggestion: diagnostics::OuterCrateLevelAttrSuggestion {
                                        bang_position,
                                    },
                                },
                            )
                        }
                        Some(ast::AttrStyle::Inner) | None => self.tcx.emit_node_span_lint(
                            UNUSED_ATTRIBUTES,
                            hir_id,
                            attr.span(),
                            diagnostics::InnerCrateLevelAttr,
                        ),
                    };
                    return;
                } else {
                    let never_needs_link = self
                        .tcx
                        .crate_types()
                        .iter()
                        .all(|kind| matches!(kind, CrateType::Rlib | CrateType::StaticLib));
                    if never_needs_link {
                        diagnostics::UnusedNote::LinkerMessagesBinaryCrateOnly
                    } else {
                        return;
                    }
                }
            } else if hir_id == CRATE_HIR_ID
                && attr.has_any_name(&[sym::allow, sym::warn, sym::deny, sym::forbid, sym::expect])
                && let Some(meta) = attr.meta_item_list()
                && meta.iter().any(|meta| {
                    meta.meta_item().is_some_and(|item| item.path == sym::dead_code_pub_in_binary)
                })
                && !self.tcx.crate_types().contains(&CrateType::Executable)
            {
                diagnostics::UnusedNote::NoEffectDeadCodePubInBinary
            } else if attr.has_name(sym::default_method_body_is_const) {
                diagnostics::UnusedNote::DefaultMethodBodyConst
            } else {
                return;
            };

        self.tcx.emit_node_span_lint(
            UNUSED_ATTRIBUTES,
            hir_id,
            attr.span(),
            diagnostics::Unused { attr_span: attr.span(), note },
        );
    }

    /// A best effort attempt to create an error for a mismatching proc macro signature.
    ///
    /// If this best effort goes wrong, it will just emit a worse error later (see #102923)
    fn check_proc_macro(&self, hir_id: HirId, target: Target, kind: ProcMacroKind) {
        if target != Target::Fn {
            return;
        }

        let tcx = self.tcx;
        let Some(token_stream_def_id) = tcx.get_diagnostic_item(sym::TokenStream) else {
            return;
        };
        let Some(token_stream) = tcx.type_of(token_stream_def_id).no_bound_vars() else {
            return;
        };

        let def_id = hir_id.expect_owner().def_id;
        let param_env = ty::ParamEnv::empty();

        let infcx = tcx.infer_ctxt().build(TypingMode::non_body_analysis());
        let ocx = ObligationCtxt::new_with_diagnostics(&infcx);

        let span = tcx.def_span(def_id);
        let fresh_args = infcx.fresh_args_for_item(span, def_id.to_def_id());
        let sig = tcx.liberate_late_bound_regions(
            def_id.to_def_id(),
            tcx.fn_sig(def_id).instantiate(tcx, fresh_args).skip_norm_wip(),
        );

        let mut cause = ObligationCause::misc(span, def_id);
        let sig = ocx.normalize(&cause, param_env, Unnormalized::new_wip(sig));

        // proc macro is not WF.
        let errors = ocx.try_evaluate_obligations();
        if !errors.is_empty() {
            return;
        }

        let expected_sig = tcx.mk_fn_sig_safe_rust_abi(
            std::iter::repeat_n(
                token_stream,
                match kind {
                    ProcMacroKind::Attribute => 2,
                    ProcMacroKind::Derive | ProcMacroKind::FunctionLike => 1,
                },
            ),
            token_stream,
        );

        if let Err(terr) = ocx.eq(&cause, param_env, expected_sig, sig) {
            let mut diag = tcx.dcx().create_err(diagnostics::ProcMacroBadSig { span, kind });

            let hir_sig = tcx.hir_fn_sig_by_hir_id(hir_id);
            if let Some(hir_sig) = hir_sig {
                match terr {
                    TypeError::ArgumentMutability(idx) | TypeError::ArgumentSorts(_, idx) => {
                        if let Some(ty) = hir_sig.decl.inputs.get(idx) {
                            diag.span(ty.span);
                            cause.span = ty.span;
                        } else if idx == hir_sig.decl.inputs.len() {
                            let span = hir_sig.decl.output.span();
                            diag.span(span);
                            cause.span = span;
                        }
                    }
                    TypeError::ArgCount => {
                        if let Some(ty) = hir_sig.decl.inputs.get(expected_sig.inputs().len()) {
                            diag.span(ty.span);
                            cause.span = ty.span;
                        }
                    }
                    TypeError::SafetyMismatch(_) => {
                        // FIXME: Would be nice if we had a span here..
                    }
                    TypeError::AbiMismatch(_) => {
                        // FIXME: Would be nice if we had a span here..
                    }
                    TypeError::VariadicMismatch(_) => {
                        // FIXME: Would be nice if we had a span here..
                    }
                    _ => {}
                }
            }

            infcx.err_ctxt().note_type_err(
                &mut diag,
                &cause,
                None,
                Some(param_env.and(ValuePairs::PolySigs(ExpectedFound {
                    expected: ty::Binder::dummy(expected_sig),
                    found: ty::Binder::dummy(sig),
                }))),
                terr,
                false,
                None,
            );
            diag.emit();
            self.abort.set(true);
        }

        let errors = ocx.evaluate_obligations_error_on_ambiguity();
        if !errors.is_empty() {
            infcx.err_ctxt().report_fulfillment_errors(errors);
            self.abort.set(true);
        }
    }

    fn check_rustc_force_inline(&self, hir_id: HirId, attrs: &[Attribute], target: Target) {
        if let (Target::Closure, None) = (
            target,
            find_attr!(attrs, Inline(InlineAttr::Force { attr_span, .. }, _) => *attr_span),
        ) {
            let is_coro = matches!(
                self.tcx.hir_expect_expr(hir_id).kind,
                hir::ExprKind::Closure(hir::Closure {
                    kind: hir::ClosureKind::Coroutine(..) | hir::ClosureKind::CoroutineClosure(..),
                    ..
                })
            );
            let parent_did = self.tcx.hir_get_parent_item(hir_id).to_def_id();
            let parent_span = self.tcx.def_span(parent_did);

            if let Some(attr_span) = find_attr!(
                self.tcx, parent_did,
                Inline(InlineAttr::Force { attr_span, .. }, _) => *attr_span
            ) && is_coro
            {
                self.dcx()
                    .emit_err(diagnostics::RustcForceInlineCoro { attr_span, span: parent_span });
            }
        }
    }

    fn check_mix_no_mangle_export(&self, hir_id: HirId, attrs: &[Attribute]) {
        if let Some(export_name_span) =
            find_attr!(attrs, ExportName { span: export_name_span, .. } => *export_name_span)
            && let Some(no_mangle_span) =
                find_attr!(attrs, NoMangle(no_mangle_span) => *no_mangle_span)
        {
            let no_mangle_attr = if no_mangle_span.edition() >= Edition::Edition2024 {
                "#[unsafe(no_mangle)]"
            } else {
                "#[no_mangle]"
            };
            let export_name_attr = if export_name_span.edition() >= Edition::Edition2024 {
                "#[unsafe(export_name)]"
            } else {
                "#[export_name]"
            };

            self.tcx.emit_node_span_lint(
                lint::builtin::UNUSED_ATTRIBUTES,
                hir_id,
                no_mangle_span,
                diagnostics::MixedExportNameAndNoMangle {
                    no_mangle_span,
                    export_name_span,
                    no_mangle_attr,
                    export_name_attr,
                },
            );
        }
    }

    fn check_optimize_and_inline(&self, attrs: &[Attribute]) {
        if let Some(optimize_span) =
            find_attr!(attrs, Optimize(OptimizeAttr::DoNotOptimize, span) => *span)
            && let Some((inline_attr, inline_span)) =
                find_attr!(attrs, Inline(inline_attr, span) => (inline_attr, *span))
            && inline_attr != &InlineAttr::Never
        {
            self.dcx()
                .emit_err(diagnostics::BothOptimizeNoneAndInline { optimize_span, inline_span });
        }
    }
}

impl<'tcx> Visitor<'tcx> for CheckAttrVisitor<'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.tcx
    }

    fn visit_item(&mut self, item: &'tcx Item<'tcx>) {
        // Historically we've run more checks on non-exported than exported macros,
        // so this lets us continue to run them while maintaining backwards compatibility.
        // In the long run, the checks should be harmonized.
        if let ItemKind::Macro(_, macro_def, _) = item.kind {
            let def_id = item.owner_id.to_def_id();
            if macro_def.macro_rules && !find_attr!(self.tcx, def_id, MacroExport { .. }) {
                check_non_exported_macro_for_invalid_attrs(self.tcx, item);
            }
        }

        let target = Target::from_item(item);
        self.check_attributes(item.hir_id(), item.span, target, Some(item));
        intravisit::walk_item(self, item)
    }

    fn visit_where_predicate(&mut self, where_predicate: &'tcx hir::WherePredicate<'tcx>) {
        self.check_attributes(
            where_predicate.hir_id,
            where_predicate.span,
            Target::WherePredicate,
            None,
        );
        intravisit::walk_where_predicate(self, where_predicate)
    }

    fn visit_generic_param(&mut self, generic_param: &'tcx hir::GenericParam<'tcx>) {
        let target = Target::from_generic_param(generic_param);
        self.check_attributes(generic_param.hir_id, generic_param.span, target, None);
        intravisit::walk_generic_param(self, generic_param)
    }

    fn visit_trait_item(&mut self, trait_item: &'tcx TraitItem<'tcx>) {
        let target = Target::from_trait_item(trait_item);
        self.check_attributes(trait_item.hir_id(), trait_item.span, target, None);
        intravisit::walk_trait_item(self, trait_item)
    }

    fn visit_field_def(&mut self, struct_field: &'tcx hir::FieldDef<'tcx>) {
        self.check_attributes(struct_field.hir_id, struct_field.span, Target::Field, None);
        intravisit::walk_field_def(self, struct_field);
    }

    fn visit_arm(&mut self, arm: &'tcx hir::Arm<'tcx>) {
        self.check_attributes(arm.hir_id, arm.span, Target::Arm, None);
        intravisit::walk_arm(self, arm);
    }

    fn visit_foreign_item(&mut self, f_item: &'tcx ForeignItem<'tcx>) {
        let target = Target::from_foreign_item(f_item);
        self.check_attributes(f_item.hir_id(), f_item.span, target, None);
        intravisit::walk_foreign_item(self, f_item)
    }

    fn visit_impl_item(&mut self, impl_item: &'tcx hir::ImplItem<'tcx>) {
        let target = target_from_impl_item(self.tcx, impl_item);
        self.check_attributes(impl_item.hir_id(), impl_item.span, target, None);
        intravisit::walk_impl_item(self, impl_item)
    }

    fn visit_stmt(&mut self, stmt: &'tcx hir::Stmt<'tcx>) {
        // When checking statements ignore expressions, they will be checked later.
        if let hir::StmtKind::Let(l) = stmt.kind {
            self.check_attributes(l.hir_id, stmt.span, Target::Statement, None);
        }
        intravisit::walk_stmt(self, stmt)
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
        let target = match expr.kind {
            hir::ExprKind::Closure { .. } => Target::Closure,
            _ => Target::Expression,
        };

        self.check_attributes(expr.hir_id, expr.span, target, None);
        intravisit::walk_expr(self, expr)
    }

    fn visit_expr_field(&mut self, field: &'tcx hir::ExprField<'tcx>) {
        self.check_attributes(field.hir_id, field.span, Target::ExprField, None);
        intravisit::walk_expr_field(self, field)
    }

    fn visit_variant(&mut self, variant: &'tcx hir::Variant<'tcx>) {
        self.check_attributes(variant.hir_id, variant.span, Target::Variant, None);
        intravisit::walk_variant(self, variant)
    }

    fn visit_param(&mut self, param: &'tcx hir::Param<'tcx>) {
        self.check_attributes(param.hir_id, param.span, Target::Param, None);

        intravisit::walk_param(self, param);
    }

    fn visit_pat_field(&mut self, field: &'tcx hir::PatField<'tcx>) {
        self.check_attributes(field.hir_id, field.span, Target::PatField, None);
        intravisit::walk_pat_field(self, field);
    }
}

fn is_c_like_enum(item: &Item<'_>) -> bool {
    if let ItemKind::Enum(_, _, ref def) = item.kind {
        for variant in def.variants {
            match variant.data {
                hir::VariantData::Unit(..) => { /* continue */ }
                _ => return false,
            }
        }
        true
    } else {
        false
    }
}

fn check_non_exported_macro_for_invalid_attrs(tcx: TyCtxt<'_>, item: &Item<'_>) {
    let attrs = tcx.hir_attrs(item.hir_id());

    if let Some(attr_span) =
        find_attr!(attrs, Inline(i, span) if !matches!(i, InlineAttr::Force{..}) => *span)
    {
        tcx.dcx().emit_err(diagnostics::NonExportedMacroInvalidAttrs { attr_span });
    }
}

fn check_mod_attrs(tcx: TyCtxt<'_>, module_def_id: LocalModId) {
    let check_attr_visitor = &mut CheckAttrVisitor { tcx, abort: Cell::new(false) };
    tcx.hir_visit_item_likes_in_module(module_def_id, check_attr_visitor);
    if module_def_id.to_local_def_id().is_top_level_module() {
        check_attr_visitor.check_attributes(CRATE_HIR_ID, DUMMY_SP, Target::Mod, None);
    }
    if check_attr_visitor.abort.get() {
        tcx.dcx().abort_if_errors()
    }
}

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers { check_mod_attrs, ..*providers };
}

fn doc_fake_variadic_is_allowed_self_ty(self_ty: &hir::Ty<'_>) -> bool {
    matches!(&self_ty.kind, hir::TyKind::Tup([_]))
        || if let hir::TyKind::FnPtr(fn_ptr_ty) = &self_ty.kind {
            fn_ptr_ty.decl.inputs.len() == 1
        } else {
            false
        }
        || (if let hir::TyKind::Path(hir::QPath::Resolved(_, path)) = &self_ty.kind
            && let Some(&[hir::GenericArg::Type(ty)]) =
                path.segments.last().map(|last| last.args().args)
        {
            doc_fake_variadic_is_allowed_self_ty(ty.as_unambig_ty())
        } else {
            false
        })
}
