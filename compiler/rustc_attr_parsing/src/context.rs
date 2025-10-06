use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ops::{Deref, DerefMut};
use std::sync::LazyLock;

use private::Sealed;
use rustc_ast::{AttrStyle, CRATE_NODE_ID, MetaItemLit, NodeId};
use rustc_errors::{Diag, Diagnostic, Level};
use rustc_feature::AttributeTemplate;
use rustc_hir::attrs::AttributeKind;
use rustc_hir::lints::{AttributeLint, AttributeLintKind};
use rustc_hir::{AttrPath, CRATE_HIR_ID, HirId};
use rustc_session::Session;
use rustc_span::{ErrorGuaranteed, Span, Symbol};

use crate::AttributeParser;
use crate::attributes::allow_unstable::{
    AllowConstFnUnstableParser, AllowInternalUnstableParser, UnstableFeatureBoundParser,
};
use crate::attributes::body::CoroutineParser;
use crate::attributes::codegen_attrs::{
    ColdParser, CoverageParser, ExportNameParser, ForceTargetFeatureParser, NakedParser,
    NoMangleParser, ObjcClassParser, ObjcSelectorParser, OptimizeParser, SanitizeParser,
    TargetFeatureParser, TrackCallerParser, UsedParser,
};
use crate::attributes::confusables::ConfusablesParser;
use crate::attributes::crate_level::{
    CrateNameParser, MoveSizeLimitParser, NoCoreParser, NoStdParser, PatternComplexityLimitParser,
    RecursionLimitParser, RustcCoherenceIsCoreParser, TypeLengthLimitParser,
};
use crate::attributes::debugger::DebuggerViualizerParser;
use crate::attributes::deprecation::DeprecationParser;
use crate::attributes::dummy::DummyParser;
use crate::attributes::inline::{InlineParser, RustcForceInlineParser};
use crate::attributes::link_attrs::{
    ExportStableParser, FfiConstParser, FfiPureParser, LinkNameParser, LinkOrdinalParser,
    LinkParser, LinkSectionParser, LinkageParser, StdInternalSymbolParser,
};
use crate::attributes::lint_helpers::{
    AsPtrParser, AutomaticallyDerivedParser, PassByValueParser, PubTransparentParser,
};
use crate::attributes::loop_match::{ConstContinueParser, LoopMatchParser};
use crate::attributes::macro_attrs::{
    AllowInternalUnsafeParser, MacroEscapeParser, MacroExportParser, MacroUseParser,
};
use crate::attributes::must_use::MustUseParser;
use crate::attributes::no_implicit_prelude::NoImplicitPreludeParser;
use crate::attributes::non_exhaustive::NonExhaustiveParser;
use crate::attributes::path::PathParser as PathAttributeParser;
use crate::attributes::proc_macro_attrs::{
    ProcMacroAttributeParser, ProcMacroDeriveParser, ProcMacroParser, RustcBuiltinMacroParser,
};
use crate::attributes::prototype::CustomMirParser;
use crate::attributes::repr::{AlignParser, AlignStaticParser, ReprParser};
use crate::attributes::rustc_internal::{
    RustcLayoutScalarValidRangeEnd, RustcLayoutScalarValidRangeStart,
    RustcObjectLifetimeDefaultParser, RustcSimdMonomorphizeLaneLimitParser,
};
use crate::attributes::semantics::MayDangleParser;
use crate::attributes::stability::{
    BodyStabilityParser, ConstStabilityIndirectParser, ConstStabilityParser, StabilityParser,
};
use crate::attributes::test_attrs::{IgnoreParser, ShouldPanicParser};
use crate::attributes::traits::{
    AllowIncoherentImplParser, CoinductiveParser, ConstTraitParser, DenyExplicitImplParser,
    DoNotImplementViaObjectParser, FundamentalParser, MarkerParser, ParenSugarParser,
    PointeeParser, SkipDuringMethodDispatchParser, SpecializationTraitParser, TypeConstParser,
    UnsafeSpecializationMarkerParser,
};
use crate::attributes::transparency::TransparencyParser;
use crate::attributes::{AttributeParser as _, Combine, Single, WithoutArgs};
use crate::parser::{ArgParser, PathParser};
use crate::session_diagnostics::{AttributeParseError, AttributeParseErrorReason, UnknownMetaItem};
use crate::target_checking::AllowedTargets;

type GroupType<S> = LazyLock<GroupTypeInner<S>>;

pub(super) struct GroupTypeInner<S: Stage> {
    pub(super) accepters: BTreeMap<&'static [Symbol], Vec<GroupTypeInnerAccept<S>>>,
    pub(super) finalizers: Vec<FinalizeFn<S>>,
}

pub(super) struct GroupTypeInnerAccept<S: Stage> {
    pub(super) template: AttributeTemplate,
    pub(super) accept_fn: AcceptFn<S>,
    pub(super) allowed_targets: AllowedTargets,
}

type AcceptFn<S> =
    Box<dyn for<'sess, 'a> Fn(&mut AcceptContext<'_, 'sess, S>, &ArgParser<'a>) + Send + Sync>;
type FinalizeFn<S> =
    Box<dyn Send + Sync + Fn(&mut FinalizeContext<'_, '_, S>) -> Option<AttributeKind>>;

macro_rules! attribute_parsers {
    (
        pub(crate) static $name: ident = [$($names: ty),* $(,)?];
    ) => {
        mod early {
            use super::*;
            type Combine<T> = super::Combine<T, Early>;
            type Single<T> = super::Single<T, Early>;
            type WithoutArgs<T> = super::WithoutArgs<T, Early>;

            attribute_parsers!(@[Early] pub(crate) static $name = [$($names),*];);
        }
        mod late {
            use super::*;
            type Combine<T> = super::Combine<T, Late>;
            type Single<T> = super::Single<T, Late>;
            type WithoutArgs<T> = super::WithoutArgs<T, Late>;

            attribute_parsers!(@[Late] pub(crate) static $name = [$($names),*];);
        }
    };
    (
        @[$stage: ty] pub(crate) static $name: ident = [$($names: ty),* $(,)?];
    ) => {
        pub(crate) static $name: GroupType<$stage> = LazyLock::new(|| {
            let mut accepts = BTreeMap::<_, Vec<GroupTypeInnerAccept<$stage>>>::new();
            let mut finalizes = Vec::<FinalizeFn<$stage>>::new();
            $(
                {
                    thread_local! {
                        static STATE_OBJECT: RefCell<$names> = RefCell::new(<$names>::default());
                    };

                    for (path, template, accept_fn) in <$names>::ATTRIBUTES {
                        accepts.entry(*path).or_default().push(GroupTypeInnerAccept {
                            template: *template,
                            accept_fn: Box::new(|cx, args| {
                                STATE_OBJECT.with_borrow_mut(|s| {
                                    accept_fn(s, cx, args)
                                })
                            }),
                            allowed_targets: <$names as crate::attributes::AttributeParser<$stage>>::ALLOWED_TARGETS,
                        });
                    }

                    finalizes.push(Box::new(|cx| {
                        let state = STATE_OBJECT.take();
                        state.finalize(cx)
                    }));
                }
            )*

            GroupTypeInner { accepters:accepts, finalizers:finalizes }
        });
    };
}
attribute_parsers!(
    pub(crate) static ATTRIBUTE_PARSERS = [
        // tidy-alphabetical-start
        AlignParser,
        AlignStaticParser,
        BodyStabilityParser,
        ConfusablesParser,
        ConstStabilityParser,
        MacroUseParser,
        NakedParser,
        StabilityParser,
        UsedParser,
        // tidy-alphabetical-end

        // tidy-alphabetical-start
        Combine<AllowConstFnUnstableParser>,
        Combine<AllowInternalUnstableParser>,
        Combine<DebuggerViualizerParser>,
        Combine<ForceTargetFeatureParser>,
        Combine<LinkParser>,
        Combine<ReprParser>,
        Combine<TargetFeatureParser>,
        Combine<UnstableFeatureBoundParser>,
        // tidy-alphabetical-end

        // tidy-alphabetical-start
        Single<CoverageParser>,
        Single<CrateNameParser>,
        Single<CustomMirParser>,
        Single<DeprecationParser>,
        Single<DummyParser>,
        Single<ExportNameParser>,
        Single<IgnoreParser>,
        Single<InlineParser>,
        Single<LinkNameParser>,
        Single<LinkOrdinalParser>,
        Single<LinkSectionParser>,
        Single<LinkageParser>,
        Single<MacroExportParser>,
        Single<MoveSizeLimitParser>,
        Single<MustUseParser>,
        Single<ObjcClassParser>,
        Single<ObjcSelectorParser>,
        Single<OptimizeParser>,
        Single<PathAttributeParser>,
        Single<PatternComplexityLimitParser>,
        Single<ProcMacroDeriveParser>,
        Single<RecursionLimitParser>,
        Single<RustcBuiltinMacroParser>,
        Single<RustcForceInlineParser>,
        Single<RustcLayoutScalarValidRangeEnd>,
        Single<RustcLayoutScalarValidRangeStart>,
        Single<RustcObjectLifetimeDefaultParser>,
        Single<RustcSimdMonomorphizeLaneLimitParser>,
        Single<SanitizeParser>,
        Single<ShouldPanicParser>,
        Single<SkipDuringMethodDispatchParser>,
        Single<TransparencyParser>,
        Single<TypeLengthLimitParser>,
        Single<WithoutArgs<AllowIncoherentImplParser>>,
        Single<WithoutArgs<AllowInternalUnsafeParser>>,
        Single<WithoutArgs<AsPtrParser>>,
        Single<WithoutArgs<AutomaticallyDerivedParser>>,
        Single<WithoutArgs<CoinductiveParser>>,
        Single<WithoutArgs<ColdParser>>,
        Single<WithoutArgs<ConstContinueParser>>,
        Single<WithoutArgs<ConstStabilityIndirectParser>>,
        Single<WithoutArgs<ConstTraitParser>>,
        Single<WithoutArgs<CoroutineParser>>,
        Single<WithoutArgs<DenyExplicitImplParser>>,
        Single<WithoutArgs<DoNotImplementViaObjectParser>>,
        Single<WithoutArgs<ExportStableParser>>,
        Single<WithoutArgs<FfiConstParser>>,
        Single<WithoutArgs<FfiPureParser>>,
        Single<WithoutArgs<FundamentalParser>>,
        Single<WithoutArgs<LoopMatchParser>>,
        Single<WithoutArgs<MacroEscapeParser>>,
        Single<WithoutArgs<MarkerParser>>,
        Single<WithoutArgs<MayDangleParser>>,
        Single<WithoutArgs<NoCoreParser>>,
        Single<WithoutArgs<NoImplicitPreludeParser>>,
        Single<WithoutArgs<NoMangleParser>>,
        Single<WithoutArgs<NoStdParser>>,
        Single<WithoutArgs<NonExhaustiveParser>>,
        Single<WithoutArgs<ParenSugarParser>>,
        Single<WithoutArgs<PassByValueParser>>,
        Single<WithoutArgs<PointeeParser>>,
        Single<WithoutArgs<ProcMacroAttributeParser>>,
        Single<WithoutArgs<ProcMacroParser>>,
        Single<WithoutArgs<PubTransparentParser>>,
        Single<WithoutArgs<RustcCoherenceIsCoreParser>>,
        Single<WithoutArgs<SpecializationTraitParser>>,
        Single<WithoutArgs<StdInternalSymbolParser>>,
        Single<WithoutArgs<TrackCallerParser>>,
        Single<WithoutArgs<TypeConstParser>>,
        Single<WithoutArgs<UnsafeSpecializationMarkerParser>>,
        // tidy-alphabetical-end
    ];
);

mod private {
    pub trait Sealed {}
    impl Sealed for super::Early {}
    impl Sealed for super::Late {}
}

// allow because it's a sealed trait
#[allow(private_interfaces)]
pub trait Stage: Sized + 'static + Sealed {
    type Id: Copy;

    fn parsers() -> &'static GroupType<Self>;

    fn emit_err<'sess>(
        &self,
        sess: &'sess Session,
        diag: impl for<'x> Diagnostic<'x>,
    ) -> ErrorGuaranteed;

    fn should_emit(&self) -> ShouldEmit;

    fn id_is_crate_root(id: Self::Id) -> bool;
}

// allow because it's a sealed trait
#[allow(private_interfaces)]
impl Stage for Early {
    type Id = NodeId;

    fn parsers() -> &'static GroupType<Self> {
        &early::ATTRIBUTE_PARSERS
    }
    fn emit_err<'sess>(
        &self,
        sess: &'sess Session,
        diag: impl for<'x> Diagnostic<'x>,
    ) -> ErrorGuaranteed {
        self.should_emit().emit_err(sess.dcx().create_err(diag))
    }

    fn should_emit(&self) -> ShouldEmit {
        self.emit_errors
    }

    fn id_is_crate_root(id: Self::Id) -> bool {
        id == CRATE_NODE_ID
    }
}

// allow because it's a sealed trait
#[allow(private_interfaces)]
impl Stage for Late {
    type Id = HirId;

    fn parsers() -> &'static GroupType<Self> {
        &late::ATTRIBUTE_PARSERS
    }
    fn emit_err<'sess>(
        &self,
        tcx: &'sess Session,
        diag: impl for<'x> Diagnostic<'x>,
    ) -> ErrorGuaranteed {
        tcx.dcx().emit_err(diag)
    }

    fn should_emit(&self) -> ShouldEmit {
        ShouldEmit::ErrorsAndLints
    }

    fn id_is_crate_root(id: Self::Id) -> bool {
        id == CRATE_HIR_ID
    }
}

/// used when parsing attributes for miscellaneous things *before* ast lowering
pub struct Early {
    /// Whether to emit errors or delay them as a bug
    /// For most attributes, the attribute will be parsed again in the `Late` stage and in this case the errors should be delayed
    /// But for some, such as `cfg`, the attribute will be removed before the `Late` stage so errors must be emitted
    pub emit_errors: ShouldEmit,
}
/// used when parsing attributes during ast lowering
pub struct Late;

/// Context given to every attribute parser when accepting
///
/// Gives [`AttributeParser`]s enough information to create errors, for example.
pub struct AcceptContext<'f, 'sess, S: Stage> {
    pub(crate) shared: SharedContext<'f, 'sess, S>,
    /// The span of the attribute currently being parsed
    pub(crate) attr_span: Span,

    /// Whether it is an inner or outer attribute
    pub(crate) attr_style: AttrStyle,

    /// The expected structure of the attribute.
    ///
    /// Used in reporting errors to give a hint to users what the attribute *should* look like.
    pub(crate) template: &'f AttributeTemplate,

    /// The name of the attribute we're currently accepting.
    pub(crate) attr_path: AttrPath,
}

impl<'f, 'sess: 'f, S: Stage> SharedContext<'f, 'sess, S> {
    pub(crate) fn emit_err(&self, diag: impl for<'x> Diagnostic<'x>) -> ErrorGuaranteed {
        self.stage.emit_err(&self.sess, diag)
    }

    /// Emit a lint. This method is somewhat special, since lints emitted during attribute parsing
    /// must be delayed until after HIR is built. This method will take care of the details of
    /// that.
    pub(crate) fn emit_lint(&mut self, lint: AttributeLintKind, span: Span) {
        if !matches!(
            self.stage.should_emit(),
            ShouldEmit::ErrorsAndLints | ShouldEmit::EarlyFatal { also_emit_lints: true }
        ) {
            return;
        }
        let id = self.target_id;
        (self.emit_lint)(AttributeLint { id, span, kind: lint });
    }

    pub(crate) fn warn_unused_duplicate(&mut self, used_span: Span, unused_span: Span) {
        self.emit_lint(
            AttributeLintKind::UnusedDuplicate {
                this: unused_span,
                other: used_span,
                warning: false,
            },
            unused_span,
        )
    }

    pub(crate) fn warn_unused_duplicate_future_error(
        &mut self,
        used_span: Span,
        unused_span: Span,
    ) {
        self.emit_lint(
            AttributeLintKind::UnusedDuplicate {
                this: unused_span,
                other: used_span,
                warning: true,
            },
            unused_span,
        )
    }
}

impl<'f, 'sess: 'f, S: Stage> AcceptContext<'f, 'sess, S> {
    pub(crate) fn unknown_key(
        &self,
        span: Span,
        found: String,
        options: &'static [&'static str],
    ) -> ErrorGuaranteed {
        self.emit_err(UnknownMetaItem { span, item: found, expected: options })
    }

    /// error that a string literal was expected.
    /// You can optionally give the literal you did find (which you found not to be a string literal)
    /// which can make better errors. For example, if the literal was a byte string it will suggest
    /// removing the `b` prefix.
    pub(crate) fn expected_string_literal(
        &self,
        span: Span,
        actual_literal: Option<&MetaItemLit>,
    ) -> ErrorGuaranteed {
        self.emit_err(AttributeParseError {
            span,
            attr_span: self.attr_span,
            template: self.template.clone(),
            attribute: self.attr_path.clone(),
            reason: AttributeParseErrorReason::ExpectedStringLiteral {
                byte_string: actual_literal.and_then(|i| {
                    i.kind.is_bytestr().then(|| self.sess().source_map().start_point(i.span))
                }),
            },
            attr_style: self.attr_style,
        })
    }

    pub(crate) fn expected_integer_literal(&self, span: Span) -> ErrorGuaranteed {
        self.emit_err(AttributeParseError {
            span,
            attr_span: self.attr_span,
            template: self.template.clone(),
            attribute: self.attr_path.clone(),
            reason: AttributeParseErrorReason::ExpectedIntegerLiteral,
            attr_style: self.attr_style,
        })
    }

    pub(crate) fn expected_list(&self, span: Span) -> ErrorGuaranteed {
        self.emit_err(AttributeParseError {
            span,
            attr_span: self.attr_span,
            template: self.template.clone(),
            attribute: self.attr_path.clone(),
            reason: AttributeParseErrorReason::ExpectedList,
            attr_style: self.attr_style,
        })
    }

    pub(crate) fn expected_no_args(&self, args_span: Span) -> ErrorGuaranteed {
        self.emit_err(AttributeParseError {
            span: args_span,
            attr_span: self.attr_span,
            template: self.template.clone(),
            attribute: self.attr_path.clone(),
            reason: AttributeParseErrorReason::ExpectedNoArgs,
            attr_style: self.attr_style,
        })
    }

    /// emit an error that a `name` was expected here
    pub(crate) fn expected_identifier(&self, span: Span) -> ErrorGuaranteed {
        self.emit_err(AttributeParseError {
            span,
            attr_span: self.attr_span,
            template: self.template.clone(),
            attribute: self.attr_path.clone(),
            reason: AttributeParseErrorReason::ExpectedIdentifier,
            attr_style: self.attr_style,
        })
    }

    /// emit an error that a `name = value` pair was expected at this span. The symbol can be given for
    /// a nicer error message talking about the specific name that was found lacking a value.
    pub(crate) fn expected_name_value(&self, span: Span, name: Option<Symbol>) -> ErrorGuaranteed {
        self.emit_err(AttributeParseError {
            span,
            attr_span: self.attr_span,
            template: self.template.clone(),
            attribute: self.attr_path.clone(),
            reason: AttributeParseErrorReason::ExpectedNameValue(name),
            attr_style: self.attr_style,
        })
    }

    /// emit an error that a `name = value` pair was found where that name was already seen.
    pub(crate) fn duplicate_key(&self, span: Span, key: Symbol) -> ErrorGuaranteed {
        self.emit_err(AttributeParseError {
            span,
            attr_span: self.attr_span,
            template: self.template.clone(),
            attribute: self.attr_path.clone(),
            reason: AttributeParseErrorReason::DuplicateKey(key),
            attr_style: self.attr_style,
        })
    }

    /// an error that should be emitted when a [`MetaItemOrLitParser`](crate::parser::MetaItemOrLitParser)
    /// was expected *not* to be a literal, but instead a meta item.
    pub(crate) fn unexpected_literal(&self, span: Span) -> ErrorGuaranteed {
        self.emit_err(AttributeParseError {
            span,
            attr_span: self.attr_span,
            template: self.template.clone(),
            attribute: self.attr_path.clone(),
            reason: AttributeParseErrorReason::UnexpectedLiteral,
            attr_style: self.attr_style,
        })
    }

    pub(crate) fn expected_single_argument(&self, span: Span) -> ErrorGuaranteed {
        self.emit_err(AttributeParseError {
            span,
            attr_span: self.attr_span,
            template: self.template.clone(),
            attribute: self.attr_path.clone(),
            reason: AttributeParseErrorReason::ExpectedSingleArgument,
            attr_style: self.attr_style,
        })
    }

    pub(crate) fn expected_at_least_one_argument(&self, span: Span) -> ErrorGuaranteed {
        self.emit_err(AttributeParseError {
            span,
            attr_span: self.attr_span,
            template: self.template.clone(),
            attribute: self.attr_path.clone(),
            reason: AttributeParseErrorReason::ExpectedAtLeastOneArgument,
            attr_style: self.attr_style,
        })
    }

    /// produces an error along the lines of `expected one of [foo, meow]`
    pub(crate) fn expected_specific_argument(
        &self,
        span: Span,
        possibilities: &[Symbol],
    ) -> ErrorGuaranteed {
        self.emit_err(AttributeParseError {
            span,
            attr_span: self.attr_span,
            template: self.template.clone(),
            attribute: self.attr_path.clone(),
            reason: AttributeParseErrorReason::ExpectedSpecificArgument {
                possibilities,
                strings: false,
                list: false,
            },
            attr_style: self.attr_style,
        })
    }

    /// produces an error along the lines of `expected one of [foo, meow] as an argument`.
    /// i.e. slightly different wording to [`expected_specific_argument`](Self::expected_specific_argument).
    pub(crate) fn expected_specific_argument_and_list(
        &self,
        span: Span,
        possibilities: &[Symbol],
    ) -> ErrorGuaranteed {
        self.emit_err(AttributeParseError {
            span,
            attr_span: self.attr_span,
            template: self.template.clone(),
            attribute: self.attr_path.clone(),
            reason: AttributeParseErrorReason::ExpectedSpecificArgument {
                possibilities,
                strings: false,
                list: true,
            },
            attr_style: self.attr_style,
        })
    }

    /// produces an error along the lines of `expected one of ["foo", "meow"]`
    pub(crate) fn expected_specific_argument_strings(
        &self,
        span: Span,
        possibilities: &[Symbol],
    ) -> ErrorGuaranteed {
        self.emit_err(AttributeParseError {
            span,
            attr_span: self.attr_span,
            template: self.template.clone(),
            attribute: self.attr_path.clone(),
            reason: AttributeParseErrorReason::ExpectedSpecificArgument {
                possibilities,
                strings: true,
                list: false,
            },
            attr_style: self.attr_style,
        })
    }

    pub(crate) fn warn_empty_attribute(&mut self, span: Span) {
        let attr_path = self.attr_path.clone();
        let valid_without_list = self.template.word;
        self.emit_lint(
            AttributeLintKind::EmptyAttribute { first_span: span, attr_path, valid_without_list },
            span,
        );
    }
}

impl<'f, 'sess, S: Stage> Deref for AcceptContext<'f, 'sess, S> {
    type Target = SharedContext<'f, 'sess, S>;

    fn deref(&self) -> &Self::Target {
        &self.shared
    }
}

impl<'f, 'sess, S: Stage> DerefMut for AcceptContext<'f, 'sess, S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.shared
    }
}

/// Context given to every attribute parser during finalization.
///
/// Gives [`AttributeParser`](crate::attributes::AttributeParser)s enough information to create
/// errors, for example.
pub struct SharedContext<'p, 'sess, S: Stage> {
    /// The parse context, gives access to the session and the
    /// diagnostics context.
    pub(crate) cx: &'p mut AttributeParser<'sess, S>,
    /// The span of the syntactical component this attribute was applied to
    pub(crate) target_span: Span,
    /// The id ([`NodeId`] if `S` is `Early`, [`HirId`] if `S` is `Late`) of the syntactical component this attribute was applied to
    pub(crate) target_id: S::Id,

    pub(crate) emit_lint: &'p mut dyn FnMut(AttributeLint<S::Id>),
}

/// Context given to every attribute parser during finalization.
///
/// Gives [`AttributeParser`](crate::attributes::AttributeParser)s enough information to create
/// errors, for example.
pub(crate) struct FinalizeContext<'p, 'sess, S: Stage> {
    pub(crate) shared: SharedContext<'p, 'sess, S>,

    /// A list of all attribute on this syntax node.
    ///
    /// Useful for compatibility checks with other attributes in [`finalize`](crate::attributes::AttributeParser::finalize)
    ///
    /// Usually, you should use normal attribute parsing logic instead,
    /// especially when making a *denylist* of other attributes.
    pub(crate) all_attrs: &'p [PathParser<'p>],
}

impl<'p, 'sess: 'p, S: Stage> Deref for FinalizeContext<'p, 'sess, S> {
    type Target = SharedContext<'p, 'sess, S>;

    fn deref(&self) -> &Self::Target {
        &self.shared
    }
}

impl<'p, 'sess: 'p, S: Stage> DerefMut for FinalizeContext<'p, 'sess, S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.shared
    }
}

impl<'p, 'sess: 'p, S: Stage> Deref for SharedContext<'p, 'sess, S> {
    type Target = AttributeParser<'sess, S>;

    fn deref(&self) -> &Self::Target {
        self.cx
    }
}

impl<'p, 'sess: 'p, S: Stage> DerefMut for SharedContext<'p, 'sess, S> {
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
    ErrorsAndLints,
    /// The operation will emit *not* errors and lints.
    /// Use this if you are *sure* that this operation will be called at a different time with `ShouldEmit::ErrorsAndLints`.
    Nothing,
}

impl ShouldEmit {
    pub(crate) fn emit_err(&self, diag: Diag<'_>) -> ErrorGuaranteed {
        match self {
            ShouldEmit::EarlyFatal { .. } if diag.level() == Level::DelayedBug => diag.emit(),
            ShouldEmit::EarlyFatal { .. } => diag.upgrade_to_fatal().emit(),
            ShouldEmit::ErrorsAndLints => diag.emit(),
            ShouldEmit::Nothing => diag.delay_as_bug(),
        }
    }
}
