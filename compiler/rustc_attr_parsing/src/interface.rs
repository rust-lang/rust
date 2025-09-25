use std::borrow::Cow;

use rustc_ast as ast;
use rustc_ast::NodeId;
use rustc_errors::DiagCtxtHandle;
use rustc_feature::{AttributeTemplate, Features};
use rustc_hir::attrs::AttributeKind;
use rustc_hir::lints::AttributeLint;
use rustc_hir::{AttrArgs, AttrItem, AttrPath, Attribute, HashIgnoredAttrId, Target};
use rustc_session::Session;
use rustc_span::{DUMMY_SP, Span, Symbol, sym};

use crate::context::{AcceptContext, FinalizeContext, SharedContext, Stage};
use crate::parser::{ArgParser, MetaItemParser, PathParser};
use crate::{Early, Late, OmitDoc, ShouldEmit};

/// Context created once, for example as part of the ast lowering
/// context, through which all attributes can be lowered.
pub struct AttributeParser<'sess, S: Stage = Late> {
    pub(crate) tools: Vec<Symbol>,
    pub(crate) features: Option<&'sess Features>,
    pub(crate) sess: &'sess Session,
    pub(crate) stage: S,

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
        features: Option<&'sess Features>,
    ) -> Option<Attribute> {
        Self::parse_limited_should_emit(
            sess,
            attrs,
            sym,
            target_span,
            target_node_id,
            features,
            ShouldEmit::Nothing,
        )
    }

    /// Usually you want `parse_limited`, which defaults to no errors.
    pub fn parse_limited_should_emit(
        sess: &'sess Session,
        attrs: &[ast::Attribute],
        sym: Symbol,
        target_span: Span,
        target_node_id: NodeId,
        features: Option<&'sess Features>,
        should_emit: ShouldEmit,
    ) -> Option<Attribute> {
        let mut parsed = Self::parse_limited_all(
            sess,
            attrs,
            Some(sym),
            Target::Crate, // Does not matter, we're not going to emit errors anyways
            target_span,
            target_node_id,
            features,
            should_emit,
        );
        assert!(parsed.len() <= 1);
        parsed.pop()
    }

    pub fn parse_limited_all(
        sess: &'sess Session,
        attrs: &[ast::Attribute],
        parse_only: Option<Symbol>,
        target: Target,
        target_span: Span,
        target_node_id: NodeId,
        features: Option<&'sess Features>,
        emit_errors: ShouldEmit,
    ) -> Vec<Attribute> {
        let mut p =
            Self { features, tools: Vec::new(), parse_only, sess, stage: Early { emit_errors } };
        p.parse_attribute_list(
            attrs,
            target_span,
            target_node_id,
            target,
            OmitDoc::Skip,
            std::convert::identity,
            |lint| {
                crate::lints::emit_attribute_lint(&lint, sess);
            },
        )
    }

    pub fn parse_single<T>(
        sess: &'sess Session,
        attr: &ast::Attribute,
        target_span: Span,
        target_node_id: NodeId,
        features: Option<&'sess Features>,
        emit_errors: ShouldEmit,
        parse_fn: fn(cx: &mut AcceptContext<'_, '_, Early>, item: &ArgParser<'_>) -> Option<T>,
        template: &AttributeTemplate,
    ) -> Option<T> {
        let mut parser = Self {
            features,
            tools: Vec::new(),
            parse_only: None,
            sess,
            stage: Early { emit_errors },
        };
        let ast::AttrKind::Normal(normal_attr) = &attr.kind else {
            panic!("parse_single called on a doc attr")
        };
        let parts =
            normal_attr.item.path.segments.iter().map(|seg| seg.ident.name).collect::<Vec<_>>();
        let meta_parser = MetaItemParser::from_attr(normal_attr, &parts, &sess.psess, emit_errors)?;
        let path = meta_parser.path();
        let args = meta_parser.args();
        let mut cx: AcceptContext<'_, 'sess, Early> = AcceptContext {
            shared: SharedContext {
                cx: &mut parser,
                target_span,
                target_id: target_node_id,
                emit_lint: &mut |lint| {
                    crate::lints::emit_attribute_lint(&lint, sess);
                },
            },
            attr_span: attr.span,
            attr_style: attr.style,
            template,
            attr_path: path.get_attribute_path(),
        };
        parse_fn(&mut cx, args)
    }
}

impl<'sess, S: Stage> AttributeParser<'sess, S> {
    pub fn new(
        sess: &'sess Session,
        features: &'sess Features,
        tools: Vec<Symbol>,
        stage: S,
    ) -> Self {
        Self { features: Some(features), tools, parse_only: None, sess, stage }
    }

    pub(crate) fn sess(&self) -> &'sess Session {
        &self.sess
    }

    pub(crate) fn features(&self) -> &'sess Features {
        self.features.expect("features not available at this point in the compiler")
    }

    pub(crate) fn features_option(&self) -> Option<&'sess Features> {
        self.features
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
        target: Target,
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
                    attr_paths.push(PathParser(Cow::Borrowed(&n.item.path)));

                    let parts =
                        n.item.path.segments.iter().map(|seg| seg.ident.name).collect::<Vec<_>>();

                    if let Some(accepts) = S::parsers().accepters.get(parts.as_slice()) {
                        let Some(parser) = MetaItemParser::from_attr(
                            n,
                            &parts,
                            &self.sess.psess,
                            self.stage.should_emit(),
                        ) else {
                            continue;
                        };
                        let path = parser.path();
                        let args = parser.args();
                        for accept in accepts {
                            let mut cx: AcceptContext<'_, 'sess, S> = AcceptContext {
                                shared: SharedContext {
                                    cx: self,
                                    target_span,
                                    target_id,
                                    emit_lint: &mut emit_lint,
                                },
                                attr_span: lower_span(attr.span),
                                attr_style: attr.style,
                                template: &accept.template,
                                attr_path: path.get_attribute_path(),
                            };

                            (accept.accept_fn)(&mut cx, args);
                            if !matches!(cx.stage.should_emit(), ShouldEmit::Nothing) {
                                Self::check_target(&accept.allowed_targets, target, &mut cx);
                            }
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
        for f in &S::parsers().finalizers {
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

    /// Returns whether there is a parser for an attribute with this name
    pub fn is_parsed_attribute(path: &[Symbol]) -> bool {
        Late::parsers().accepters.contains_key(path)
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
