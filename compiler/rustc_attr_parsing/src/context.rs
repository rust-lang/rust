use std::cell::RefCell;
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::LazyLock;

use private::Sealed;
use rustc_ast::{self as ast, MetaItemLit, NodeId};
use rustc_attr_data_structures::AttributeKind;
use rustc_attr_data_structures::lints::{AttributeLint, AttributeLintKind};
use rustc_errors::{DiagCtxtHandle, Diagnostic};
use rustc_feature::{AttributeTemplate, Features};
use rustc_hir::{AttrArgs, AttrItem, AttrPath, Attribute, HashIgnoredAttrId, HirId};
use rustc_session::Session;
use rustc_span::{DUMMY_SP, ErrorGuaranteed, Span, Symbol, sym};

use crate::attributes::allow_unstable::{AllowConstFnUnstableParser, AllowInternalUnstableParser};
use crate::attributes::codegen_attrs::{
    ColdParser, ExportNameParser, NakedParser, NoMangleParser, OptimizeParser, TrackCallerParser,
    UsedParser,
};
use crate::attributes::confusables::ConfusablesParser;
use crate::attributes::deprecation::DeprecationParser;
use crate::attributes::inline::{InlineParser, RustcForceInlineParser};
use crate::attributes::link_attrs::{LinkNameParser, LinkSectionParser};
use crate::attributes::lint_helpers::{AsPtrParser, PubTransparentParser};
use crate::attributes::loop_match::{ConstContinueParser, LoopMatchParser};
use crate::attributes::must_use::MustUseParser;
use crate::attributes::repr::{AlignParser, ReprParser};
use crate::attributes::rustc_internal::{
    RustcLayoutScalarValidRangeEnd, RustcLayoutScalarValidRangeStart,
    RustcObjectLifetimeDefaultParser,
};
use crate::attributes::semantics::MayDangleParser;
use crate::attributes::stability::{
    BodyStabilityParser, ConstStabilityIndirectParser, ConstStabilityParser, StabilityParser,
};
use crate::attributes::traits::SkipDuringMethodDispatchParser;
use crate::attributes::transparency::TransparencyParser;
use crate::attributes::{AttributeParser as _, Combine, Single};
use crate::parser::{ArgParser, MetaItemParser, PathParser};
use crate::session_diagnostics::{AttributeParseError, AttributeParseErrorReason, UnknownMetaItem};

macro_rules! group_type {
    ($stage: ty) => {
         LazyLock<(
            BTreeMap<&'static [Symbol], Vec<(AttributeTemplate, Box<dyn for<'sess, 'a> Fn(&mut AcceptContext<'_, 'sess, $stage>, &ArgParser<'a>) + Send + Sync>)>>,
            Vec<Box<dyn Send + Sync + Fn(&mut FinalizeContext<'_, '_, $stage>) -> Option<AttributeKind>>>
        )>
    };
}

macro_rules! attribute_parsers {
    (
        pub(crate) static $name: ident = [$($names: ty),* $(,)?];
    ) => {
        mod early {
            use super::*;
            type Combine<T> = super::Combine<T, Early>;
            type Single<T> = super::Single<T, Early>;

            attribute_parsers!(@[Early] pub(crate) static $name = [$($names),*];);
        }
        mod late {
            use super::*;
            type Combine<T> = super::Combine<T, Late>;
            type Single<T> = super::Single<T, Late>;

            attribute_parsers!(@[Late] pub(crate) static $name = [$($names),*];);
        }
    };
    (
        @[$ty: ty] pub(crate) static $name: ident = [$($names: ty),* $(,)?];
    ) => {
        pub(crate) static $name: group_type!($ty) = LazyLock::new(|| {
            let mut accepts = BTreeMap::<_, Vec<(AttributeTemplate, Box<dyn for<'sess, 'a> Fn(&mut AcceptContext<'_, 'sess, $ty>, &ArgParser<'a>) + Send + Sync>)>>::new();
            let mut finalizes = Vec::<Box<dyn Send + Sync + Fn(&mut FinalizeContext<'_, '_, $ty>) -> Option<AttributeKind>>>::new();
            $(
                {
                    thread_local! {
                        static STATE_OBJECT: RefCell<$names> = RefCell::new(<$names>::default());
                    };

                    for (path, template, accept_fn) in <$names>::ATTRIBUTES {
                        accepts.entry(*path).or_default().push((*template, Box::new(|cx, args| {
                            STATE_OBJECT.with_borrow_mut(|s| {
                                accept_fn(s, cx, args)
                            })
                        })));
                    }

                    finalizes.push(Box::new(|cx| {
                        let state = STATE_OBJECT.take();
                        state.finalize(cx)
                    }));
                }
            )*

            (accepts, finalizes)
        });
    };
}
attribute_parsers!(
    pub(crate) static ATTRIBUTE_PARSERS = [
        // tidy-alphabetical-start
        AlignParser,
        BodyStabilityParser,
        ConfusablesParser,
        ConstStabilityParser,
        NakedParser,
        StabilityParser,
        UsedParser,
        // tidy-alphabetical-end

        // tidy-alphabetical-start
        Combine<AllowConstFnUnstableParser>,
        Combine<AllowInternalUnstableParser>,
        Combine<ReprParser>,
        // tidy-alphabetical-end

        // tidy-alphabetical-start
        Single<AsPtrParser>,
        Single<ColdParser>,
        Single<ConstContinueParser>,
        Single<ConstStabilityIndirectParser>,
        Single<DeprecationParser>,
        Single<ExportNameParser>,
        Single<InlineParser>,
        Single<LinkNameParser>,
        Single<LinkSectionParser>,
        Single<LoopMatchParser>,
        Single<MayDangleParser>,
        Single<MustUseParser>,
        Single<NoMangleParser>,
        Single<OptimizeParser>,
        Single<PubTransparentParser>,
        Single<RustcForceInlineParser>,
        Single<RustcLayoutScalarValidRangeEnd>,
        Single<RustcLayoutScalarValidRangeStart>,
        Single<RustcObjectLifetimeDefaultParser>,
        Single<SkipDuringMethodDispatchParser>,
        Single<TrackCallerParser>,
        Single<TransparencyParser>,
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

    fn parsers() -> &'static group_type!(Self);

    fn emit_err<'sess>(sess: &'sess Session, diag: impl for<'x> Diagnostic<'x>) -> ErrorGuaranteed;
}

// allow because it's a sealed trait
#[allow(private_interfaces)]
impl Stage for Early {
    type Id = NodeId;

    fn parsers() -> &'static group_type!(Self) {
        &early::ATTRIBUTE_PARSERS
    }
    fn emit_err<'sess>(sess: &'sess Session, diag: impl for<'x> Diagnostic<'x>) -> ErrorGuaranteed {
        sess.dcx().create_err(diag).delay_as_bug()
    }
}

// allow because it's a sealed trait
#[allow(private_interfaces)]
impl Stage for Late {
    type Id = HirId;

    fn parsers() -> &'static group_type!(Self) {
        &late::ATTRIBUTE_PARSERS
    }
    fn emit_err<'sess>(tcx: &'sess Session, diag: impl for<'x> Diagnostic<'x>) -> ErrorGuaranteed {
        tcx.dcx().emit_err(diag)
    }
}

/// used when parsing attributes for miscelaneous things *before* ast lowering
pub struct Early;
/// used when parsing attributes during ast lowering
pub struct Late;

/// Context given to every attribute parser when accepting
///
/// Gives [`AttributeParser`]s enough information to create errors, for example.
pub(crate) struct AcceptContext<'f, 'sess, S: Stage> {
    pub(crate) shared: SharedContext<'f, 'sess, S>,
    /// The span of the attribute currently being parsed
    pub(crate) attr_span: Span,

    /// The expected structure of the attribute.
    ///
    /// Used in reporting errors to give a hint to users what the attribute *should* look like.
    pub(crate) template: &'f AttributeTemplate,

    /// The name of the attribute we're currently accepting.
    pub(crate) attr_path: AttrPath,
}

impl<'f, 'sess: 'f, S: Stage> SharedContext<'f, 'sess, S> {
    pub(crate) fn emit_err(&self, diag: impl for<'x> Diagnostic<'x>) -> ErrorGuaranteed {
        S::emit_err(&self.sess, diag)
    }

    /// Emit a lint. This method is somewhat special, since lints emitted during attribute parsing
    /// must be delayed until after HIR is built. This method will take care of the details of
    /// that.
    pub(crate) fn emit_lint(&mut self, lint: AttributeLintKind, span: Span) {
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
        })
    }

    pub(crate) fn expected_integer_literal(&self, span: Span) -> ErrorGuaranteed {
        self.emit_err(AttributeParseError {
            span,
            attr_span: self.attr_span,
            template: self.template.clone(),
            attribute: self.attr_path.clone(),
            reason: AttributeParseErrorReason::ExpectedIntegerLiteral,
        })
    }

    pub(crate) fn expected_list(&self, span: Span) -> ErrorGuaranteed {
        self.emit_err(AttributeParseError {
            span,
            attr_span: self.attr_span,
            template: self.template.clone(),
            attribute: self.attr_path.clone(),
            reason: AttributeParseErrorReason::ExpectedList,
        })
    }

    pub(crate) fn expected_no_args(&self, args_span: Span) -> ErrorGuaranteed {
        self.emit_err(AttributeParseError {
            span: args_span,
            attr_span: self.attr_span,
            template: self.template.clone(),
            attribute: self.attr_path.clone(),
            reason: AttributeParseErrorReason::ExpectedNoArgs,
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
        })
    }

    pub(crate) fn expected_single_argument(&self, span: Span) -> ErrorGuaranteed {
        self.emit_err(AttributeParseError {
            span,
            attr_span: self.attr_span,
            template: self.template.clone(),
            attribute: self.attr_path.clone(),
            reason: AttributeParseErrorReason::ExpectedSingleArgument,
        })
    }

    pub(crate) fn expected_at_least_one_argument(&self, span: Span) -> ErrorGuaranteed {
        self.emit_err(AttributeParseError {
            span,
            attr_span: self.attr_span,
            template: self.template.clone(),
            attribute: self.attr_path.clone(),
            reason: AttributeParseErrorReason::ExpectedAtLeastOneArgument,
        })
    }

    pub(crate) fn expected_specific_argument(
        &self,
        span: Span,
        possibilities: Vec<&'static str>,
    ) -> ErrorGuaranteed {
        self.emit_err(AttributeParseError {
            span,
            attr_span: self.attr_span,
            template: self.template.clone(),
            attribute: self.attr_path.clone(),
            reason: AttributeParseErrorReason::ExpectedSpecificArgument {
                possibilities,
                strings: false,
            },
        })
    }

    pub(crate) fn expected_specific_argument_strings(
        &self,
        span: Span,
        possibilities: Vec<&'static str>,
    ) -> ErrorGuaranteed {
        self.emit_err(AttributeParseError {
            span,
            attr_span: self.attr_span,
            template: self.template.clone(),
            attribute: self.attr_path.clone(),
            reason: AttributeParseErrorReason::ExpectedSpecificArgument {
                possibilities,
                strings: true,
            },
        })
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
pub(crate) struct SharedContext<'p, 'sess, S: Stage> {
    /// The parse context, gives access to the session and the
    /// diagnostics context.
    pub(crate) cx: &'p mut AttributeParser<'sess, S>,
    /// The span of the syntactical component this attribute was applied to
    pub(crate) target_span: Span,
    /// The id ([`NodeId`] if `S` is `Early`, [`HirId`] if `S` is `Late`) of the syntactical component this attribute was applied to
    pub(crate) target_id: S::Id,

    emit_lint: &'p mut dyn FnMut(AttributeLint<S::Id>),
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

/// Context created once, for example as part of the ast lowering
/// context, through which all attributes can be lowered.
pub struct AttributeParser<'sess, S: Stage = Late> {
    pub(crate) tools: Vec<Symbol>,
    features: Option<&'sess Features>,
    sess: &'sess Session,
    stage: PhantomData<S>,

    /// *Only* parse attributes with this symbol.
    ///
    /// Used in cases where we want the lowering infrastructure for parse just a single attribute.
    parse_only: Option<Symbol>,
}

impl<'sess> AttributeParser<'sess, Early> {
    /// This method allows you to parse attributes *before* you have access to features or tools.
    /// One example where this is necessary, is to parse `feature` attributes themselves for
    /// example.
    ///
    /// Try to use this as little as possible. Attributes *should* be lowered during
    /// `rustc_ast_lowering`. Some attributes require access to features to parse, which would
    /// crash if you tried to do so through [`parse_limited`](Self::parse_limited).
    ///
    /// To make sure use is limited, supply a `Symbol` you'd like to parse. Only attributes with
    /// that symbol are picked out of the list of instructions and parsed. Those are returned.
    ///
    /// No diagnostics will be emitted when parsing limited. Lints are not emitted at all, while
    /// errors will be emitted as a delayed bugs. in other words, we *expect* attributes parsed
    /// with `parse_limited` to be reparsed later during ast lowering where we *do* emit the errors
    pub fn parse_limited(
        sess: &'sess Session,
        attrs: &[ast::Attribute],
        sym: Symbol,
        target_span: Span,
        target_node_id: NodeId,
    ) -> Option<Attribute> {
        let mut p = Self {
            features: None,
            tools: Vec::new(),
            parse_only: Some(sym),
            sess,
            stage: PhantomData,
        };
        let mut parsed = p.parse_attribute_list(
            attrs,
            target_span,
            target_node_id,
            OmitDoc::Skip,
            std::convert::identity,
            |_lint| {
                panic!("can't emit lints here for now (nothing uses this atm)");
            },
        );
        assert!(parsed.len() <= 1);

        parsed.pop()
    }
}

impl<'sess, S: Stage> AttributeParser<'sess, S> {
    pub fn new(sess: &'sess Session, features: &'sess Features, tools: Vec<Symbol>) -> Self {
        Self { features: Some(features), tools, parse_only: None, sess, stage: PhantomData }
    }

    pub(crate) fn sess(&self) -> &'sess Session {
        &self.sess
    }

    pub(crate) fn features(&self) -> &'sess Features {
        self.features.expect("features not available at this point in the compiler")
    }

    pub(crate) fn dcx(&self) -> DiagCtxtHandle<'sess> {
        self.sess().dcx()
    }

    /// Parse a list of attributes.
    ///
    /// `target_span` is the span of the thing this list of attributes is applied to,
    /// and when `omit_doc` is set, doc attributes are filtered out.
    pub fn parse_attribute_list(
        &mut self,
        attrs: &[ast::Attribute],
        target_span: Span,
        target_id: S::Id,
        omit_doc: OmitDoc,

        lower_span: impl Copy + Fn(Span) -> Span,
        mut emit_lint: impl FnMut(AttributeLint<S::Id>),
    ) -> Vec<Attribute> {
        let mut attributes = Vec::new();
        let mut attr_paths = Vec::new();

        for attr in attrs {
            // If we're only looking for a single attribute, skip all the ones we don't care about.
            if let Some(expected) = self.parse_only {
                if !attr.has_name(expected) {
                    continue;
                }
            }

            // Sometimes, for example for `#![doc = include_str!("readme.md")]`,
            // doc still contains a non-literal. You might say, when we're lowering attributes
            // that's expanded right? But no, sometimes, when parsing attributes on macros,
            // we already use the lowering logic and these are still there. So, when `omit_doc`
            // is set we *also* want to ignore these.
            if omit_doc == OmitDoc::Skip && attr.has_name(sym::doc) {
                continue;
            }

            match &attr.kind {
                ast::AttrKind::DocComment(comment_kind, symbol) => {
                    if omit_doc == OmitDoc::Skip {
                        continue;
                    }

                    attributes.push(Attribute::Parsed(AttributeKind::DocComment {
                        style: attr.style,
                        kind: *comment_kind,
                        span: lower_span(attr.span),
                        comment: *symbol,
                    }))
                }
                // // FIXME: make doc attributes go through a proper attribute parser
                // ast::AttrKind::Normal(n) if n.has_name(sym::doc) => {
                //     let p = GenericMetaItemParser::from_attr(&n, self.dcx());
                //
                //     attributes.push(Attribute::Parsed(AttributeKind::DocComment {
                //         style: attr.style,
                //         kind: CommentKind::Line,
                //         span: attr.span,
                //         comment: p.args().name_value(),
                //     }))
                // }
                ast::AttrKind::Normal(n) => {
                    attr_paths.push(PathParser::Ast(&n.item.path));

                    let parser = MetaItemParser::from_attr(n, self.dcx());
                    let path = parser.path();
                    let args = parser.args();
                    let parts = path.segments().map(|i| i.name).collect::<Vec<_>>();

                    if let Some(accepts) = S::parsers().0.get(parts.as_slice()) {
                        for (template, accept) in accepts {
                            let mut cx: AcceptContext<'_, 'sess, S> = AcceptContext {
                                shared: SharedContext {
                                    cx: self,
                                    target_span,
                                    target_id,
                                    emit_lint: &mut emit_lint,
                                },
                                attr_span: lower_span(attr.span),
                                template,
                                attr_path: path.get_attribute_path(),
                            };

                            accept(&mut cx, args)
                        }
                    } else {
                        // If we're here, we must be compiling a tool attribute... Or someone
                        // forgot to parse their fancy new attribute. Let's warn them in any case.
                        // If you are that person, and you really think your attribute should
                        // remain unparsed, carefully read the documentation in this module and if
                        // you still think so you can add an exception to this assertion.

                        // FIXME(jdonszelmann): convert other attributes, and check with this that
                        // we caught em all
                        // const FIXME_TEMPORARY_ATTR_ALLOWLIST: &[Symbol] = &[sym::cfg];
                        // assert!(
                        //     self.tools.contains(&parts[0]) || true,
                        //     // || FIXME_TEMPORARY_ATTR_ALLOWLIST.contains(&parts[0]),
                        //     "attribute {path} wasn't parsed and isn't a know tool attribute",
                        // );

                        attributes.push(Attribute::Unparsed(Box::new(AttrItem {
                            path: AttrPath::from_ast(&n.item.path),
                            args: self.lower_attr_args(&n.item.args, lower_span),
                            id: HashIgnoredAttrId { attr_id: attr.id },
                            style: attr.style,
                            span: lower_span(attr.span),
                        })));
                    }
                }
            }
        }

        let mut parsed_attributes = Vec::new();
        for f in &S::parsers().1 {
            if let Some(attr) = f(&mut FinalizeContext {
                shared: SharedContext {
                    cx: self,
                    target_span,
                    target_id,
                    emit_lint: &mut emit_lint,
                },
                all_attrs: &attr_paths,
            }) {
                parsed_attributes.push(Attribute::Parsed(attr));
            }
        }

        attributes.extend(parsed_attributes);

        attributes
    }

    fn lower_attr_args(&self, args: &ast::AttrArgs, lower_span: impl Fn(Span) -> Span) -> AttrArgs {
        match args {
            ast::AttrArgs::Empty => AttrArgs::Empty,
            ast::AttrArgs::Delimited(args) => AttrArgs::Delimited(args.clone()),
            // This is an inert key-value attribute - it will never be visible to macros
            // after it gets lowered to HIR. Therefore, we can extract literals to handle
            // nonterminals in `#[doc]` (e.g. `#[doc = $e]`).
            ast::AttrArgs::Eq { eq_span, expr } => {
                // In valid code the value always ends up as a single literal. Otherwise, a dummy
                // literal suffices because the error is handled elsewhere.
                let lit = if let ast::ExprKind::Lit(token_lit) = expr.kind
                    && let Ok(lit) =
                        ast::MetaItemLit::from_token_lit(token_lit, lower_span(expr.span))
                {
                    lit
                } else {
                    let guar = self.dcx().span_delayed_bug(
                        args.span().unwrap_or(DUMMY_SP),
                        "expr in place where literal is expected (builtin attr parsing)",
                    );
                    ast::MetaItemLit {
                        symbol: sym::dummy,
                        suffix: None,
                        kind: ast::LitKind::Err(guar),
                        span: DUMMY_SP,
                    }
                };
                AttrArgs::Eq { eq_span: lower_span(*eq_span), expr: lit }
            }
        }
    }
}
