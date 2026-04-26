use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::sync::LazyLock;

use rustc_ast::{AttrStyle, MetaItemLit};
use rustc_data_structures::sync::{DynSend, DynSync};
use rustc_errors::{Diag, DiagCtxtHandle, Diagnostic, Level, MultiSpan};
use rustc_feature::{AttrSuggestionStyle, AttributeTemplate};
use rustc_hir::AttrPath;
use rustc_hir::attrs::AttributeKind;
use rustc_hir::lints::AttributeLintKind;
use rustc_parse::parser::Recovery;
use rustc_session::lint::{Lint, LintId};
use rustc_span::{ErrorGuaranteed, Span, Symbol};

// Glob imports to avoid big, bitrotty import lists
use crate::attributes::allow_unstable::*;
use crate::attributes::autodiff::*;
use crate::attributes::body::*;
use crate::attributes::cfi_encoding::*;
use crate::attributes::codegen_attrs::*;
use crate::attributes::confusables::*;
use crate::attributes::crate_level::*;
use crate::attributes::debugger::*;
use crate::attributes::deprecation::*;
use crate::attributes::diagnostic::do_not_recommend::*;
use crate::attributes::diagnostic::on_const::*;
use crate::attributes::diagnostic::on_move::*;
use crate::attributes::diagnostic::on_unimplemented::*;
use crate::attributes::diagnostic::on_unknown::*;
use crate::attributes::diagnostic::on_unmatch_args::*;
use crate::attributes::doc::*;
use crate::attributes::dummy::*;
use crate::attributes::inline::*;
use crate::attributes::instruction_set::*;
use crate::attributes::link_attrs::*;
use crate::attributes::lint_helpers::*;
use crate::attributes::loop_match::*;
use crate::attributes::macro_attrs::*;
use crate::attributes::must_not_suspend::*;
use crate::attributes::must_use::*;
use crate::attributes::no_implicit_prelude::*;
use crate::attributes::no_link::*;
use crate::attributes::non_exhaustive::*;
use crate::attributes::path::PathParser as PathAttributeParser;
use crate::attributes::pin_v2::*;
use crate::attributes::proc_macro_attrs::*;
use crate::attributes::prototype::*;
use crate::attributes::repr::*;
use crate::attributes::rustc_allocator::*;
use crate::attributes::rustc_dump::*;
use crate::attributes::rustc_internal::*;
use crate::attributes::semantics::*;
use crate::attributes::stability::*;
use crate::attributes::test_attrs::*;
use crate::attributes::traits::*;
use crate::attributes::transparency::*;
use crate::attributes::{AttributeParser as _, AttributeSafety, Combine, Single, WithoutArgs};
use crate::parser::{ArgParser, MetaItemListParser, MetaItemOrLitParser, RefPathParser};
use crate::session_diagnostics::{
    AttributeParseError, AttributeParseErrorReason, AttributeParseErrorSuggestions,
    ParsedDescription,
};
use crate::target_checking::AllowedTargets;
use crate::{AttributeParser, EmitAttribute};

type GroupType = LazyLock<GroupTypeInner>;

pub(super) struct GroupTypeInner {
    pub(super) accepters: BTreeMap<&'static [Symbol], GroupTypeInnerAccept>,
}

pub(super) struct GroupTypeInnerAccept {
    pub(super) template: AttributeTemplate,
    pub(super) accept_fn: AcceptFn,
    pub(super) allowed_targets: AllowedTargets,
    pub(super) safety: AttributeSafety,
    pub(super) finalizer: FinalizeFn,
}

pub(crate) type AcceptFn = Box<
    dyn for<'sess, 'a> Fn(&mut AttributeParsers, &mut AcceptContext<'_, 'sess>, &ArgParser)
        + Send
        + Sync,
>;
pub(crate) type FinalizeFn = Box<
    dyn Send
        + Sync
        + Fn(&mut AttributeParsers, &mut FinalizeContext<'_, '_>) -> Option<AttributeKind>,
>;

macro_rules! attribute_parsers {
    (
        pub(crate) static $name: ident = [$($pname: ident: $ptype: ty),* $(,)?];
    ) => {
        #[derive(Default)]
        pub(crate) struct AttributeParsers {
            $(
                $pname: Option<$ptype>,
            )*
        }

        pub(crate) static $name: GroupType = LazyLock::new(|| {
            let mut accepters = BTreeMap::<_, GroupTypeInnerAccept>::new();
            $(
                {
                    for (path, template, accept_fn) in <$ptype>::ATTRIBUTES {
                        match accepters.entry(*path) {
                            Entry::Vacant(e) => {
                                e.insert(GroupTypeInnerAccept {
                                    template: *template,
                                    accept_fn: Box::new(|parsers, cx, args| {
                                        accept_fn(&mut parsers.$pname.get_or_insert_default(), cx, args)
                                    }),
                                    safety: <$ptype as crate::attributes::AttributeParser>::SAFETY,
                                    allowed_targets: <$ptype as crate::attributes::AttributeParser>::ALLOWED_TARGETS,
                                    finalizer: Box::new(|parsers, cx| {
                                        parsers.$pname.take().and_then(|a| a.finalize(cx))
                                    })
                                });
                            }
                            Entry::Occupied(_) => panic!("Attribute {path:?} has multiple accepters"),
                        }
                    }
                }
            )*

            GroupTypeInner { accepters }
        });
    };
}
attribute_parsers!(
    pub(crate) static ATTRIBUTE_PARSERS = [
        // tidy-alphabetical-start
        p0: BodyStabilityParser,
        p1: ConfusablesParser,
        p2: ConstStabilityParser,
        p3: DocParser,
        p4: MacroUseParser,
        p5: NakedParser,
        p6: OnConstParser,
        p7: OnMoveParser,
        p8: OnUnimplementedParser,
        p9: OnUnknownParser,
        p10: OnUnmatchArgsParser,
        p11: RustcAlignParser,
        p12: RustcAlignStaticParser,
        p13: RustcCguTestAttributeParser,
        p14: StabilityParser,
        p15: UsedParser,
        // tidy-alphabetical-end

        // tidy-alphabetical-start
        c0: Combine<AllowInternalUnstableParser>,
        c1: Combine<CrateTypeParser>,
        c2: Combine<DebuggerViualizerParser>,
        c3: Combine<FeatureParser>,
        c4: Combine<ForceTargetFeatureParser>,
        c5: Combine<LinkParser>,
        c6: Combine<RegisterToolParser>,
        c7: Combine<ReprParser>,
        c8: Combine<RustcAllowConstFnUnstableParser>,
        c9: Combine<RustcCleanParser>,
        c10: Combine<RustcDumpLayoutParser>,
        c11: Combine<RustcMirParser>,
        c12: Combine<RustcThenThisWouldNeedParser>,
        c13: Combine<TargetFeatureParser>,
        c14: Combine<UnstableFeatureBoundParser>,
        c15: Combine<UnstableRemovedParser>,
        // tidy-alphabetical-end

        // tidy-alphabetical-start
        s0: Single<CfiEncodingParser>,
        s1: Single<CollapseDebugInfoParser>,
        s2: Single<CoverageParser>,
        s3: Single<CrateNameParser>,
        s4: Single<CustomMirParser>,
        s5: Single<DeprecatedParser>,
        s6: Single<DoNotRecommendParser>,
        s7: Single<ExportNameParser>,
        s8: Single<IgnoreParser>,
        s9: Single<InlineParser>,
        s10: Single<InstructionSetParser>,
        s11: Single<LangParser>,
        s12: Single<LinkNameParser>,
        s13: Single<LinkOrdinalParser>,
        s14: Single<LinkSectionParser>,
        s15: Single<LinkageParser>,
        s16: Single<MacroExportParser>,
        s17: Single<MoveSizeLimitParser>,
        s18: Single<MustNotSuspendParser>,
        s19: Single<MustUseParser>,
        s20: Single<OptimizeParser>,
        s21: Single<PatchableFunctionEntryParser>,
        s22: Single<PathAttributeParser>,
        s23: Single<PatternComplexityLimitParser>,
        s24: Single<ProcMacroDeriveParser>,
        s25: Single<RecursionLimitParser>,
        s26: Single<ReexportTestHarnessMainParser>,
        s27: Single<RustcAbiParser>,
        s28: Single<RustcAllocatorZeroedVariantParser>,
        s29: Single<RustcAutodiffParser>,
        s30: Single<RustcBuiltinMacroParser>,
        s31: Single<RustcDeprecatedSafe2024Parser>,
        s32: Single<RustcDiagnosticItemParser>,
        s33: Single<RustcDocPrimitiveParser>,
        s34: Single<RustcDummyParser>,
        s35: Single<RustcDumpDefPathParser>,
        s36: Single<RustcDumpSymbolNameParser>,
        s37: Single<RustcForceInlineParser>,
        s38: Single<RustcIfThisChangedParser>,
        s39: Single<RustcLayoutScalarValidRangeEndParser>,
        s40: Single<RustcLayoutScalarValidRangeStartParser>,
        s41: Single<RustcLegacyConstGenericsParser>,
        s42: Single<RustcLintOptDenyFieldAccessParser>,
        s43: Single<RustcMacroTransparencyParser>,
        s44: Single<RustcMustImplementOneOfParser>,
        s45: Single<RustcNeverTypeOptionsParser>,
        s46: Single<RustcObjcClassParser>,
        s47: Single<RustcObjcSelectorParser>,
        s48: Single<RustcReservationImplParser>,
        s49: Single<RustcScalableVectorParser>,
        s50: Single<RustcSimdMonomorphizeLaneLimitParser>,
        s51: Single<RustcSkipDuringMethodDispatchParser>,
        s52: Single<RustcTestMarkerParser>,
        s53: Single<SanitizeParser>,
        s54: Single<ShouldPanicParser>,
        s55: Single<TestRunnerParser>,
        s56: Single<TypeLengthLimitParser>,
        s57: Single<WindowsSubsystemParser>,
        s58: Single<WithoutArgs<AllowInternalUnsafeParser>>,
        s59: Single<WithoutArgs<AutomaticallyDerivedParser>>,
        s60: Single<WithoutArgs<ColdParser>>,
        s61: Single<WithoutArgs<CompilerBuiltinsParser>>,
        s62: Single<WithoutArgs<ConstContinueParser>>,
        s63: Single<WithoutArgs<CoroutineParser>>,
        s64: Single<WithoutArgs<DefaultLibAllocatorParser>>,
        s65: Single<WithoutArgs<ExportStableParser>>,
        s66: Single<WithoutArgs<FfiConstParser>>,
        s67: Single<WithoutArgs<FfiPureParser>>,
        s68: Single<WithoutArgs<FundamentalParser>>,
        s69: Single<WithoutArgs<LoopMatchParser>>,
        s70: Single<WithoutArgs<MacroEscapeParser>>,
        s71: Single<WithoutArgs<MarkerParser>>,
        s72: Single<WithoutArgs<MayDangleParser>>,
        s73: Single<WithoutArgs<NeedsAllocatorParser>>,
        s74: Single<WithoutArgs<NeedsPanicRuntimeParser>>,
        s75: Single<WithoutArgs<NoBuiltinsParser>>,
        s76: Single<WithoutArgs<NoCoreParser>>,
        s77: Single<WithoutArgs<NoImplicitPreludeParser>>,
        s78: Single<WithoutArgs<NoLinkParser>>,
        s79: Single<WithoutArgs<NoMainParser>>,
        s80: Single<WithoutArgs<NoMangleParser>>,
        s81: Single<WithoutArgs<NoStdParser>>,
        s82: Single<WithoutArgs<NonExhaustiveParser>>,
        s83: Single<WithoutArgs<PanicHandlerParser>>,
        s84: Single<WithoutArgs<PanicRuntimeParser>>,
        s85: Single<WithoutArgs<PinV2Parser>>,
        s86: Single<WithoutArgs<PreludeImportParser>>,
        s87: Single<WithoutArgs<ProcMacroAttributeParser>>,
        s88: Single<WithoutArgs<ProcMacroParser>>,
        s89: Single<WithoutArgs<ProfilerRuntimeParser>>,
        s90: Single<WithoutArgs<RustcAllocatorParser>>,
        s91: Single<WithoutArgs<RustcAllocatorZeroedParser>>,
        s92: Single<WithoutArgs<RustcAllowIncoherentImplParser>>,
        s93: Single<WithoutArgs<RustcAsPtrParser>>,
        s94: Single<WithoutArgs<RustcCaptureAnalysisParser>>,
        s95: Single<WithoutArgs<RustcCoherenceIsCoreParser>>,
        s96: Single<WithoutArgs<RustcCoinductiveParser>>,
        s97: Single<WithoutArgs<RustcConstStableIndirectParser>>,
        s98: Single<WithoutArgs<RustcConversionSuggestionParser>>,
        s99: Single<WithoutArgs<RustcDeallocatorParser>>,
        s100: Single<WithoutArgs<RustcDelayedBugFromInsideQueryParser>>,
        s101: Single<WithoutArgs<RustcDenyExplicitImplParser>>,
        s102: Single<WithoutArgs<RustcDoNotConstCheckParser>>,
        s103: Single<WithoutArgs<RustcDumpDefParentsParser>>,
        s104: Single<WithoutArgs<RustcDumpHiddenTypeOfOpaquesParser>>,
        s105: Single<WithoutArgs<RustcDumpInferredOutlivesParser>>,
        s106: Single<WithoutArgs<RustcDumpItemBoundsParser>>,
        s107: Single<WithoutArgs<RustcDumpObjectLifetimeDefaultsParser>>,
        s108: Single<WithoutArgs<RustcDumpPredicatesParser>>,
        s109: Single<WithoutArgs<RustcDumpUserArgsParser>>,
        s110: Single<WithoutArgs<RustcDumpVariancesOfOpaquesParser>>,
        s111: Single<WithoutArgs<RustcDumpVariancesParser>>,
        s112: Single<WithoutArgs<RustcDumpVtableParser>>,
        s113: Single<WithoutArgs<RustcDynIncompatibleTraitParser>>,
        s114: Single<WithoutArgs<RustcEffectiveVisibilityParser>>,
        s115: Single<WithoutArgs<RustcEiiForeignItemParser>>,
        s116: Single<WithoutArgs<RustcEvaluateWhereClausesParser>>,
        s117: Single<WithoutArgs<RustcExhaustiveParser>>,
        s118: Single<WithoutArgs<RustcHasIncoherentInherentImplsParser>>,
        s119: Single<WithoutArgs<RustcInheritOverflowChecksParser>>,
        s120: Single<WithoutArgs<RustcInsignificantDtorParser>>,
        s121: Single<WithoutArgs<RustcIntrinsicConstStableIndirectParser>>,
        s122: Single<WithoutArgs<RustcIntrinsicParser>>,
        s123: Single<WithoutArgs<RustcLintOptTyParser>>,
        s124: Single<WithoutArgs<RustcLintQueryInstabilityParser>>,
        s125: Single<WithoutArgs<RustcLintUntrackedQueryInformationParser>>,
        s126: Single<WithoutArgs<RustcMainParser>>,
        s127: Single<WithoutArgs<RustcNeverReturnsNullPtrParser>>,
        s128: Single<WithoutArgs<RustcNoImplicitAutorefsParser>>,
        s129: Single<WithoutArgs<RustcNoImplicitBoundsParser>>,
        s130: Single<WithoutArgs<RustcNoMirInlineParser>>,
        s131: Single<WithoutArgs<RustcNoWritableParser>>,
        s132: Single<WithoutArgs<RustcNonConstTraitMethodParser>>,
        s133: Single<WithoutArgs<RustcNonnullOptimizationGuaranteedParser>>,
        s134: Single<WithoutArgs<RustcNounwindParser>>,
        s135: Single<WithoutArgs<RustcOffloadKernelParser>>,
        s136: Single<WithoutArgs<RustcParenSugarParser>>,
        s137: Single<WithoutArgs<RustcPassByValueParser>>,
        s138: Single<WithoutArgs<RustcPassIndirectlyInNonRusticAbisParser>>,
        s139: Single<WithoutArgs<RustcPreserveUbChecksParser>>,
        s140: Single<WithoutArgs<RustcProcMacroDeclsParser>>,
        s141: Single<WithoutArgs<RustcPubTransparentParser>>,
        s142: Single<WithoutArgs<RustcReallocatorParser>>,
        s143: Single<WithoutArgs<RustcRegionsParser>>,
        s144: Single<WithoutArgs<RustcShouldNotBeCalledOnConstItemsParser>>,
        s145: Single<WithoutArgs<RustcSpecializationTraitParser>>,
        s146: Single<WithoutArgs<RustcStdInternalSymbolParser>>,
        s147: Single<WithoutArgs<RustcStrictCoherenceParser>>,
        s148: Single<WithoutArgs<RustcTrivialFieldReadsParser>>,
        s149: Single<WithoutArgs<RustcUnsafeSpecializationMarkerParser>>,
        s150: Single<WithoutArgs<ThreadLocalParser>>,
        s151: Single<WithoutArgs<TrackCallerParser>>,
        // tidy-alphabetical-end
    ];
);

/// Context given to every attribute parser when accepting
///
/// Gives [`AttributeParser`]s enough information to create errors, for example.
pub struct AcceptContext<'f, 'sess> {
    pub(crate) shared: SharedContext<'f, 'sess>,

    /// The outer span of the attribute currently being parsed
    ///
    /// ```none
    /// #[attribute(...)]
    /// ^^^^^^^^^^^^^^^^^ outer span
    /// ```
    /// For attributes in `cfg_attr`, the outer span and inner spans are equal.
    pub(crate) attr_span: Span,
    /// The inner span of the attribute currently being parsed.
    ///
    /// ```none
    /// #[attribute(...)]
    ///   ^^^^^^^^^^^^^^  inner span
    /// ```
    pub(crate) inner_span: Span,

    /// Whether it is an inner or outer attribute.
    pub(crate) attr_style: AttrStyle,

    /// A description of the thing we are parsing using this attribute parser.
    /// We are not only using these parsers for attributes, but also for macros such as the `cfg!()` macro.
    pub(crate) parsed_description: ParsedDescription,

    /// The expected structure of the attribute.
    ///
    /// Used in reporting errors to give a hint to users what the attribute *should* look like.
    pub(crate) template: &'f AttributeTemplate,

    /// The name of the attribute we're currently accepting.
    pub(crate) attr_path: AttrPath,
}

impl<'f, 'sess: 'f> SharedContext<'f, 'sess> {
    pub(crate) fn emit_err(&self, diag: impl for<'x> Diagnostic<'x>) -> ErrorGuaranteed {
        self.cx.emit_err(diag)
    }

    /// Emit a lint. This method is somewhat special, since lints emitted during attribute parsing
    /// must be delayed until after HIR is built. This method will take care of the details of
    /// that.
    pub(crate) fn emit_lint(
        &mut self,
        lint: &'static Lint,
        kind: AttributeLintKind,
        span: impl Into<MultiSpan>,
    ) {
        self.emit_lint_inner(lint, EmitAttribute::Static(kind), span);
    }

    /// Emit a lint. This method is somewhat special, since lints emitted during attribute parsing
    /// must be delayed until after HIR is built. This method will take care of the details of
    /// that.
    pub(crate) fn emit_dyn_lint<
        F: for<'a> Fn(DiagCtxtHandle<'a>, Level) -> Diag<'a, ()> + DynSend + DynSync + 'static,
    >(
        &mut self,
        lint: &'static Lint,
        callback: F,
        span: impl Into<MultiSpan>,
    ) {
        self.emit_lint_inner(lint, EmitAttribute::Dynamic(Box::new(callback)), span);
    }

    fn emit_lint_inner(
        &mut self,
        lint: &'static Lint,
        kind: EmitAttribute,
        span: impl Into<MultiSpan>,
    ) {
        if !matches!(
            self.should_emit,
            ShouldEmit::ErrorsAndLints { .. } | ShouldEmit::EarlyFatal { also_emit_lints: true }
        ) {
            return;
        }
        (self.emit_lint)(LintId::of(lint), span.into(), kind);
    }

    pub(crate) fn warn_unused_duplicate(&mut self, used_span: Span, unused_span: Span) {
        self.emit_dyn_lint(
            rustc_session::lint::builtin::UNUSED_ATTRIBUTES,
            move |dcx, level| {
                rustc_errors::lints::UnusedDuplicate {
                    this: unused_span,
                    other: used_span,
                    warning: false,
                }
                .into_diag(dcx, level)
            },
            unused_span,
        )
    }

    pub(crate) fn warn_unused_duplicate_future_error(
        &mut self,
        used_span: Span,
        unused_span: Span,
    ) {
        self.emit_dyn_lint(
            rustc_session::lint::builtin::UNUSED_ATTRIBUTES,
            move |dcx, level| {
                rustc_errors::lints::UnusedDuplicate {
                    this: unused_span,
                    other: used_span,
                    warning: true,
                }
                .into_diag(dcx, level)
            },
            unused_span,
        )
    }
}

impl<'f, 'sess: 'f> AcceptContext<'f, 'sess> {
    pub(crate) fn adcx(&mut self) -> AttributeDiagnosticContext<'_, 'f, 'sess> {
        AttributeDiagnosticContext { ctx: self, custom_suggestions: Vec::new() }
    }

    /// Asserts that this MetaItem is a list that contains a single element. Emits an error and
    /// returns `None` if it is not the case.
    ///
    /// Some examples:
    ///
    /// - In `#[allow(warnings)]`, `warnings` is returned
    /// - In `#[cfg_attr(docsrs, doc = "foo")]`, `None` is returned, "expected a single argument
    ///   here" is emitted.
    /// - In `#[cfg()]`, `None` is returned, "expected an argument here" is emitted.
    ///
    /// The provided span is used as a fallback for diagnostic generation in case `arg` does not
    /// contain any. It should be the span of the node that contains `arg`.
    pub(crate) fn expect_single_element_list<'arg>(
        &mut self,
        arg: &'arg ArgParser,
        span: Span,
    ) -> Option<&'arg MetaItemOrLitParser> {
        let ArgParser::List(l) = arg else {
            self.adcx().expected_list(span, arg);
            return None;
        };

        let Some(single) = l.as_single() else {
            self.adcx().expected_single_argument(l.span, l.len());
            return None;
        };

        Some(single)
    }

    /// Asserts that an [`ArgParser`] is a list and returns it, or emits an error and returns
    /// `None`.
    ///
    /// Some examples:
    ///
    /// - `#[allow(clippy::complexity)]`: `(clippy::complexity)` is a list
    /// - `#[rustfmt::skip::macros(target_macro_name)]`: `(target_macro_name)` is a list
    ///
    /// This is a higher-level (and harder to misuse) wrapper over [`ArgParser::as_list`]. That
    /// allows using `?` when the attribute parsing function allows it. You may still want to use
    /// [`ArgParser::as_list`] for the following reasons:
    ///
    /// - You want to emit your own diagnostics (for instance, with [`SharedContext::emit_err`]).
    /// - The attribute can be parsed in multiple ways and it does not make sense to emit an error.
    pub(crate) fn expect_list<'arg>(
        &mut self,
        args: &'arg ArgParser,
        span: Span,
    ) -> Option<&'arg MetaItemListParser> {
        let list = args.as_list();
        if list.is_none() {
            self.adcx().expected_list(span, args);
        }
        list
    }

    /// Asserts that a [`MetaItemListParser`] contains a single element and returns it, or emits an
    /// error and returns `None`.
    ///
    /// This is a higher-level (and harder to misuse) wrapper over [`MetaItemListParser::as_single`].
    /// That allows using `?` to early return. You may still want to use
    /// [`MetaItemListParser::as_single`] for the following reasons:
    ///
    /// - You want to emit your own diagnostics (for instance, with [`SharedContext::emit_err`]).
    /// - The attribute can be parsed in multiple ways and it does not make sense to emit an error.
    pub(crate) fn expect_single<'arg>(
        &mut self,
        list: &'arg MetaItemListParser,
    ) -> Option<&'arg MetaItemOrLitParser> {
        let single = list.as_single();
        if single.is_none() {
            self.adcx().expected_single_argument(list.span, list.len());
        }
        single
    }
}

impl<'f, 'sess> Deref for AcceptContext<'f, 'sess> {
    type Target = SharedContext<'f, 'sess>;

    fn deref(&self) -> &Self::Target {
        &self.shared
    }
}

impl<'f, 'sess> DerefMut for AcceptContext<'f, 'sess> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.shared
    }
}

/// Context given to every attribute parser during finalization.
///
/// Gives [`AttributeParser`](crate::attributes::AttributeParser)s enough information to create
/// errors, for example.
pub struct SharedContext<'p, 'sess> {
    /// The parse context, gives access to the session and the
    /// diagnostics context.
    pub(crate) cx: &'p mut AttributeParser<'sess>,
    /// The span of the syntactical component this attribute was applied to
    pub(crate) target_span: Span,
    pub(crate) target: rustc_hir::Target,

    pub(crate) emit_lint: &'p mut dyn FnMut(LintId, MultiSpan, EmitAttribute),
}

/// Context given to every attribute parser during finalization.
///
/// Gives [`AttributeParser`](crate::attributes::AttributeParser)s enough information to create
/// errors, for example.
pub(crate) struct FinalizeContext<'p, 'sess> {
    pub(crate) shared: SharedContext<'p, 'sess>,

    /// A list of all attribute on this syntax node.
    ///
    /// Useful for compatibility checks with other attributes in [`finalize`](crate::attributes::AttributeParser::finalize)
    ///
    /// Usually, you should use normal attribute parsing logic instead,
    /// especially when making a *denylist* of other attributes.
    pub(crate) all_attrs: &'p [RefPathParser<'p>],
}

impl<'p, 'sess: 'p> Deref for FinalizeContext<'p, 'sess> {
    type Target = SharedContext<'p, 'sess>;

    fn deref(&self) -> &Self::Target {
        &self.shared
    }
}

impl<'p, 'sess: 'p> DerefMut for FinalizeContext<'p, 'sess> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.shared
    }
}

impl<'p, 'sess: 'p> Deref for SharedContext<'p, 'sess> {
    type Target = AttributeParser<'sess>;

    fn deref(&self) -> &Self::Target {
        self.cx
    }
}

impl<'p, 'sess: 'p> DerefMut for SharedContext<'p, 'sess> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.cx
    }
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum OmitDoc {
    Lower,
    Skip,
}

#[derive(Copy, Clone, Debug)]
pub enum ShouldEmit {
    /// The operations will emit errors, and lints, and errors are fatal.
    ///
    /// Only relevant when early parsing, in late parsing equivalent to `ErrorsAndLints`.
    /// Late parsing is never fatal, and instead tries to emit as many diagnostics as possible.
    EarlyFatal { also_emit_lints: bool },
    /// The operation will emit errors and lints.
    /// This is usually what you need.
    ErrorsAndLints {
        /// Whether [`ArgParser`] will attempt to recover from errors.
        ///
        /// Whether it is allowed to recover from bad input (like an invalid literal). Setting
        /// this to `Forbidden` will instead return early, and not raise errors except at the top
        /// level (in [`ArgParser::from_attr_args`]).
        recovery: Recovery,
    },
    /// The operation will *not* emit errors and lints.
    ///
    /// The parser can still call `delay_bug`, so you *must* ensure that this operation will also be
    /// called with `ShouldEmit::ErrorsAndLints`.
    Nothing,
}

impl ShouldEmit {
    pub(crate) fn emit_err(&self, diag: Diag<'_>) -> ErrorGuaranteed {
        match self {
            ShouldEmit::EarlyFatal { .. } if diag.level() == Level::DelayedBug => diag.emit(),
            ShouldEmit::EarlyFatal { .. } => diag.upgrade_to_fatal().emit(),
            ShouldEmit::ErrorsAndLints { .. } => diag.emit(),
            ShouldEmit::Nothing => diag.delay_as_bug(),
        }
    }
}

pub(crate) struct AttributeDiagnosticContext<'a, 'f, 'sess> {
    ctx: &'a mut AcceptContext<'f, 'sess>,
    custom_suggestions: Vec<Suggestion>,
}

impl<'a, 'f, 'sess: 'f> AttributeDiagnosticContext<'a, 'f, 'sess> {
    fn emit_parse_error(
        &mut self,
        span: Span,
        reason: AttributeParseErrorReason<'_>,
    ) -> ErrorGuaranteed {
        let suggestions = if !self.custom_suggestions.is_empty() {
            AttributeParseErrorSuggestions::CreatedByParser(mem::take(&mut self.custom_suggestions))
        } else {
            AttributeParseErrorSuggestions::CreatedByTemplate(self.template_suggestions())
        };

        self.emit_err(AttributeParseError {
            span,
            attr_span: self.attr_span,
            template: self.template.clone(),
            path: self.attr_path.clone(),
            description: self.parsed_description,
            reason,
            suggestions,
        })
    }

    /// Adds a custom suggestion to the diagnostic. This also prevents the default (template-based)
    /// suggestion to be emitted.
    pub(crate) fn push_suggestion(&mut self, msg: String, span: Span, code: String) -> &mut Self {
        self.custom_suggestions.push(Suggestion { msg, sp: span, code });
        self
    }

    pub(crate) fn template_suggestions(&self) -> Vec<String> {
        let style = match self.parsed_description {
            // If the outer and inner spans are equal, we are parsing an embedded attribute
            ParsedDescription::Attribute if self.attr_span == self.inner_span => {
                AttrSuggestionStyle::EmbeddedAttribute
            }
            ParsedDescription::Attribute => AttrSuggestionStyle::Attribute(self.attr_style),
            ParsedDescription::Macro => AttrSuggestionStyle::Macro,
        };

        self.template.suggestions(style, &self.attr_path)
    }
}

/// Helpers that can be used to generate errors during attribute parsing.
impl<'a, 'f, 'sess: 'f> AttributeDiagnosticContext<'a, 'f, 'sess> {
    pub(crate) fn expected_integer_literal_in_range(
        &mut self,
        span: Span,
        lower_bound: isize,
        upper_bound: isize,
    ) -> ErrorGuaranteed {
        self.emit_parse_error(
            span,
            AttributeParseErrorReason::ExpectedIntegerLiteralInRange { lower_bound, upper_bound },
        )
    }

    /// The provided span is used as a fallback in case `args` does not contain any. It should be
    /// the span of the node that contains `args`.
    pub(crate) fn expected_list(&mut self, span: Span, args: &ArgParser) -> ErrorGuaranteed {
        let span = match args {
            ArgParser::NoArgs => span,
            ArgParser::List(list) => list.span,
            ArgParser::NameValue(nv) => nv.args_span(),
        };
        self.emit_parse_error(span, AttributeParseErrorReason::ExpectedList)
    }

    pub(crate) fn expected_list_with_num_args_or_more(
        &mut self,
        args: usize,
        span: Span,
    ) -> ErrorGuaranteed {
        self.emit_parse_error(
            span,
            AttributeParseErrorReason::ExpectedListWithNumArgsOrMore { args },
        )
    }

    pub(crate) fn expected_list_or_no_args(&mut self, span: Span) -> ErrorGuaranteed {
        self.emit_parse_error(span, AttributeParseErrorReason::ExpectedListOrNoArgs)
    }

    pub(crate) fn expected_nv_or_no_args(&mut self, span: Span) -> ErrorGuaranteed {
        self.emit_parse_error(span, AttributeParseErrorReason::ExpectedNameValueOrNoArgs)
    }

    pub(crate) fn expected_non_empty_string_literal(&mut self, span: Span) -> ErrorGuaranteed {
        self.emit_parse_error(span, AttributeParseErrorReason::ExpectedNonEmptyStringLiteral)
    }

    pub(crate) fn expected_no_args(&mut self, span: Span) -> ErrorGuaranteed {
        self.emit_parse_error(span, AttributeParseErrorReason::ExpectedNoArgs)
    }

    /// Emit an error that a `name` was expected here
    pub(crate) fn expected_identifier(&mut self, span: Span) -> ErrorGuaranteed {
        self.emit_parse_error(span, AttributeParseErrorReason::ExpectedIdentifier)
    }

    /// Emit an error that a `name = value` pair was expected at this span. The symbol can be given for
    /// a nicer error message talking about the specific name that was found lacking a value.
    pub(crate) fn expected_name_value(
        &mut self,
        span: Span,
        name: Option<Symbol>,
    ) -> ErrorGuaranteed {
        self.emit_parse_error(span, AttributeParseErrorReason::ExpectedNameValue(name))
    }

    /// Emit an error that a `name = value` argument is missing in a list of name-value pairs.
    pub(crate) fn missing_name_value(&mut self, span: Span, name: Symbol) -> ErrorGuaranteed {
        self.emit_parse_error(span, AttributeParseErrorReason::MissingNameValue(name))
    }

    /// Emit an error that a `name = value` pair was found where that name was already seen.
    pub(crate) fn duplicate_key(&mut self, span: Span, key: Symbol) -> ErrorGuaranteed {
        self.emit_parse_error(span, AttributeParseErrorReason::DuplicateKey(key))
    }

    /// An error that should be emitted when a [`MetaItemOrLitParser`]
    /// was expected *not* to be a literal, but instead a meta item.
    pub(crate) fn expected_not_literal(&mut self, span: Span) -> ErrorGuaranteed {
        self.emit_parse_error(span, AttributeParseErrorReason::ExpectedNotLiteral)
    }

    /// Signals that we expected exactly one argument and that we got either zero or two or more.
    /// The `provided_arguments` argument allows distinguishing between "expected an argument here"
    /// (when zero arguments are provided) and "expect a single argument here" (when two or more
    /// arguments are provided).
    pub(crate) fn expected_single_argument(
        &mut self,
        span: Span,
        provided_arguments: usize,
    ) -> ErrorGuaranteed {
        let reason = if provided_arguments == 0 {
            AttributeParseErrorReason::ExpectedArgument
        } else {
            AttributeParseErrorReason::ExpectedSingleArgument
        };

        self.emit_parse_error(span, reason)
    }

    pub(crate) fn expected_at_least_one_argument(&mut self, span: Span) -> ErrorGuaranteed {
        self.emit_parse_error(span, AttributeParseErrorReason::ExpectedAtLeastOneArgument)
    }

    /// Produces an error along the lines of `expected one of [foo, meow]`
    pub(crate) fn expected_specific_argument(
        &mut self,
        span: Span,
        possibilities: &[Symbol],
    ) -> ErrorGuaranteed {
        self.emit_parse_error(
            span,
            AttributeParseErrorReason::ExpectedSpecificArgument {
                possibilities,
                strings: false,
                list: false,
            },
        )
    }

    /// Produces an error along the lines of `expected one of [foo, meow] as an argument`.
    /// i.e. slightly different wording to [`expected_specific_argument`](Self::expected_specific_argument).
    pub(crate) fn expected_specific_argument_and_list(
        &mut self,
        span: Span,
        possibilities: &[Symbol],
    ) -> ErrorGuaranteed {
        self.emit_parse_error(
            span,
            AttributeParseErrorReason::ExpectedSpecificArgument {
                possibilities,
                strings: false,
                list: true,
            },
        )
    }

    /// produces an error along the lines of `expected one of ["foo", "meow"]`
    pub(crate) fn expected_specific_argument_strings(
        &mut self,
        span: Span,
        possibilities: &[Symbol],
    ) -> ErrorGuaranteed {
        self.emit_parse_error(
            span,
            AttributeParseErrorReason::ExpectedSpecificArgument {
                possibilities,
                strings: true,
                list: false,
            },
        )
    }

    pub(crate) fn warn_empty_attribute(&mut self, span: Span) {
        let attr_path = self.attr_path.to_string();
        let valid_without_list = self.template.word;
        self.emit_dyn_lint(
            rustc_session::lint::builtin::UNUSED_ATTRIBUTES,
            move |dcx, level| {
                crate::errors::EmptyAttributeList {
                    attr_span: span,
                    attr_path: &attr_path,
                    valid_without_list,
                }
                .into_diag(dcx, level)
            },
            span,
        );
    }

    pub(crate) fn warn_ill_formed_attribute_input(&mut self, lint: &'static Lint) {
        self.warn_ill_formed_attribute_input_with_help(lint, None)
    }
    pub(crate) fn warn_ill_formed_attribute_input_with_help(
        &mut self,
        lint: &'static Lint,
        help: Option<String>,
    ) {
        let suggestions = self.suggestions();
        let span = self.attr_span;
        self.emit_dyn_lint(
            lint,
            move |dcx, level| {
                crate::errors::IllFormedAttributeInput::new(&suggestions, None, help.as_deref())
                    .into_diag(dcx, level)
            },
            span,
        );
    }

    pub(crate) fn suggestions(&self) -> Vec<String> {
        let style = match self.parsed_description {
            // If the outer and inner spans are equal, we are parsing an embedded attribute
            ParsedDescription::Attribute if self.attr_span == self.inner_span => {
                AttrSuggestionStyle::EmbeddedAttribute
            }
            ParsedDescription::Attribute => AttrSuggestionStyle::Attribute(self.attr_style),
            ParsedDescription::Macro => AttrSuggestionStyle::Macro,
        };

        self.template.suggestions(style, &self.attr_path)
    }
    /// Error that a string literal was expected.
    /// You can optionally give the literal you did find (which you found not to be a string literal)
    /// which can make better errors. For example, if the literal was a byte string it will suggest
    /// removing the `b` prefix.
    pub(crate) fn expected_string_literal(
        &mut self,
        span: Span,
        actual_literal: Option<&MetaItemLit>,
    ) -> ErrorGuaranteed {
        self.emit_parse_error(
            span,
            AttributeParseErrorReason::ExpectedStringLiteral {
                byte_string: actual_literal.and_then(|i| {
                    i.kind.is_bytestr().then(|| self.sess().source_map().start_point(i.span))
                }),
            },
        )
    }

    /// Error that a filename string literal was expected.
    pub(crate) fn expected_filename_literal(&mut self, span: Span) {
        self.emit_parse_error(span, AttributeParseErrorReason::ExpectedFilenameLiteral);
    }

    pub(crate) fn expected_integer_literal(&mut self, span: Span) -> ErrorGuaranteed {
        self.emit_parse_error(span, AttributeParseErrorReason::ExpectedIntegerLiteral)
    }
}

impl<'a, 'f, 'sess: 'f> Deref for AttributeDiagnosticContext<'a, 'f, 'sess> {
    type Target = AcceptContext<'f, 'sess>;

    fn deref(&self) -> &Self::Target {
        self.ctx
    }
}

impl<'a, 'f, 'sess: 'f> DerefMut for AttributeDiagnosticContext<'a, 'f, 'sess> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.ctx
    }
}

/// Represents a custom suggestion that an attribute parser can emit.
pub(crate) struct Suggestion {
    pub(crate) msg: String,
    pub(crate) sp: Span,
    pub(crate) code: String,
}
