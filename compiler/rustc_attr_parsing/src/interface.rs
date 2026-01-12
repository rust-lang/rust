use std::convert::identity;

use rustc_ast as ast;
use rustc_ast::token::DocFragmentKind;
use rustc_ast::{AttrItemKind, AttrStyle, NodeId, Safety};
use rustc_errors::DiagCtxtHandle;
use rustc_feature::{AttributeTemplate, Features};
use rustc_hir::attrs::AttributeKind;
use rustc_hir::lints::AttributeLint;
use rustc_hir::{AttrArgs, AttrItem, AttrPath, Attribute, HashIgnoredAttrId, Target};
use rustc_session::Session;
use rustc_session::lint::BuiltinLintDiag;
use rustc_span::{DUMMY_SP, Span, Symbol, sym};

use crate::context::{AcceptContext, FinalizeContext, SharedContext, Stage};
use crate::early_parsed::{EARLY_PARSED_ATTRIBUTES, EarlyParsedState};
use crate::parser::{ArgParser, PathParser, RefPathParser};
use crate::session_diagnostics::ParsedDescription;
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

    /// This does the same as `parse_limited`, except it has a `should_emit` parameter which allows it to emit errors.
    /// Usually you want `parse_limited`, which emits no errors.
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

    /// This method allows you to parse a list of attributes *before* `rustc_ast_lowering`.
    /// This can be used for attributes that would be removed before `rustc_ast_lowering`, such as attributes on macro calls.
    ///
    /// Try to use this as little as possible. Attributes *should* be lowered during
    /// `rustc_ast_lowering`. Some attributes require access to features to parse, which would
    /// crash if you tried to do so through [`parse_limited_all`](Self::parse_limited_all).
    /// Therefore, if `parse_only` is None, then features *must* be provided.
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
                sess.psess.buffer_lint(
                    lint.lint_id.lint,
                    lint.span,
                    lint.id,
                    BuiltinLintDiag::AttributeLint(lint.kind),
                )
            },
        )
    }

    /// This method parses a single attribute, using `parse_fn`.
    /// This is useful if you already know what exact attribute this is, and want to parse it.
    pub fn parse_single<T>(
        sess: &'sess Session,
        attr: &ast::Attribute,
        target_span: Span,
        target_node_id: NodeId,
        features: Option<&'sess Features>,
        emit_errors: ShouldEmit,
        parse_fn: fn(cx: &mut AcceptContext<'_, '_, Early>, item: &ArgParser) -> Option<T>,
        template: &AttributeTemplate,
    ) -> Option<T> {
        let ast::AttrKind::Normal(normal_attr) = &attr.kind else {
            panic!("parse_single called on a doc attr")
        };
        let parts =
            normal_attr.item.path.segments.iter().map(|seg| seg.ident.name).collect::<Vec<_>>();

        let path = AttrPath::from_ast(&normal_attr.item.path, identity);
        let args = ArgParser::from_attr_args(
            &normal_attr.item.args.unparsed_ref().unwrap(),
            &parts,
            &sess.psess,
            emit_errors,
        )?;
        Self::parse_single_args(
            sess,
            attr.span,
            normal_attr.item.span(),
            attr.style,
            path,
            Some(normal_attr.item.unsafety),
            ParsedDescription::Attribute,
            target_span,
            target_node_id,
            features,
            emit_errors,
            &args,
            parse_fn,
            template,
        )
    }

    /// This method is equivalent to `parse_single`, but parses arguments using `parse_fn` using manually created `args`.
    /// This is useful when you want to parse other things than attributes using attribute parsers.
    pub fn parse_single_args<T, I>(
        sess: &'sess Session,
        attr_span: Span,
        inner_span: Span,
        attr_style: AttrStyle,
        attr_path: AttrPath,
        attr_safety: Option<Safety>,
        parsed_description: ParsedDescription,
        target_span: Span,
        target_node_id: NodeId,
        features: Option<&'sess Features>,
        emit_errors: ShouldEmit,
        args: &I,
        parse_fn: fn(cx: &mut AcceptContext<'_, '_, Early>, item: &I) -> T,
        template: &AttributeTemplate,
    ) -> T {
        let mut parser = Self {
            features,
            tools: Vec::new(),
            parse_only: None,
            sess,
            stage: Early { emit_errors },
        };
        let mut emit_lint = |lint: AttributeLint<NodeId>| {
            sess.psess.buffer_lint(
                lint.lint_id.lint,
                lint.span,
                lint.id,
                BuiltinLintDiag::AttributeLint(lint.kind),
            )
        };
        if let Some(safety) = attr_safety {
            parser.check_attribute_safety(
                &attr_path,
                inner_span,
                safety,
                &mut emit_lint,
                target_node_id,
            )
        }
        let mut cx: AcceptContext<'_, 'sess, Early> = AcceptContext {
            shared: SharedContext {
                cx: &mut parser,
                target_span,
                target_id: target_node_id,
                emit_lint: &mut emit_lint,
            },
            attr_span,
            inner_span,
            attr_style,
            parsed_description,
            template,
            attr_path,
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
        let mut attr_paths: Vec<RefPathParser<'_>> = Vec::new();
        let mut early_parsed_state = EarlyParsedState::default();

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
            let is_doc_attribute = attr.has_name(sym::doc);
            if omit_doc == OmitDoc::Skip && is_doc_attribute {
                continue;
            }

            let attr_span = lower_span(attr.span);
            match &attr.kind {
                ast::AttrKind::DocComment(comment_kind, symbol) => {
                    if omit_doc == OmitDoc::Skip {
                        continue;
                    }

                    attributes.push(Attribute::Parsed(AttributeKind::DocComment {
                        style: attr.style,
                        kind: DocFragmentKind::Sugared(*comment_kind),
                        span: attr_span,
                        comment: *symbol,
                    }))
                }
                ast::AttrKind::Normal(n) => {
                    attr_paths.push(PathParser(&n.item.path));
                    let attr_path = AttrPath::from_ast(&n.item.path, lower_span);

                    let args = match &n.item.args {
                        AttrItemKind::Unparsed(args) => args,
                        AttrItemKind::Parsed(parsed) => {
                            early_parsed_state
                                .accept_early_parsed_attribute(attr_span, lower_span, parsed);
                            continue;
                        }
                    };

                    self.check_attribute_safety(
                        &attr_path,
                        lower_span(n.item.span()),
                        n.item.unsafety,
                        &mut emit_lint,
                        target_id,
                    );

                    let parts =
                        n.item.path.segments.iter().map(|seg| seg.ident.name).collect::<Vec<_>>();

                    if let Some(accepts) = S::parsers().accepters.get(parts.as_slice()) {
                        let Some(args) = ArgParser::from_attr_args(
                            args,
                            &parts,
                            &self.sess.psess,
                            self.stage.should_emit(),
                        ) else {
                            continue;
                        };

                        // Special-case handling for `#[doc = "..."]`: if we go through with
                        // `DocParser`, the order of doc comments will be messed up because `///`
                        // doc comments are added into `attributes` whereas attributes parsed with
                        // `DocParser` are added into `parsed_attributes` which are then appended
                        // to `attributes`. So if you have:
                        //
                        // /// bla
                        // #[doc = "a"]
                        // /// blob
                        //
                        // You would get:
                        //
                        // bla
                        // blob
                        // a
                        if is_doc_attribute
                            && let ArgParser::NameValue(nv) = &args
                            // If not a string key/value, it should emit an error, but to make
                            // things simpler, it's handled in `DocParser` because it's simpler to
                            // emit an error with `AcceptContext`.
                            && let Some(comment) = nv.value_as_str()
                        {
                            attributes.push(Attribute::Parsed(AttributeKind::DocComment {
                                style: attr.style,
                                kind: DocFragmentKind::Raw(nv.value_span),
                                span: attr_span,
                                comment,
                            }));
                            continue;
                        }

                        for accept in accepts {
                            let mut cx: AcceptContext<'_, 'sess, S> = AcceptContext {
                                shared: SharedContext {
                                    cx: self,
                                    target_span,
                                    target_id,
                                    emit_lint: &mut emit_lint,
                                },
                                attr_span,
                                inner_span: lower_span(n.item.span()),
                                attr_style: attr.style,
                                parsed_description: ParsedDescription::Attribute,
                                template: &accept.template,
                                attr_path: attr_path.clone(),
                            };

                            (accept.accept_fn)(&mut cx, &args);
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
                            path: attr_path.clone(),
                            args: self
                                .lower_attr_args(n.item.args.unparsed_ref().unwrap(), lower_span),
                            id: HashIgnoredAttrId { attr_id: attr.id },
                            style: attr.style,
                            span: attr_span,
                        })));
                    }
                }
            }
        }

        early_parsed_state.finalize_early_parsed_attributes(&mut attributes);
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
                attributes.push(Attribute::Parsed(attr));
            }
        }

        attributes
    }

    /// Returns whether there is a parser for an attribute with this name
    pub fn is_parsed_attribute(path: &[Symbol]) -> bool {
        /// The list of attributes that are parsed attributes,
        /// even though they don't have a parser in `Late::parsers()`
        const SPECIAL_ATTRIBUTES: &[&[Symbol]] = &[
            // Cfg attrs are removed after being early-parsed, so don't need to be in the parser list
            &[sym::cfg],
            &[sym::cfg_attr],
        ];

        Late::parsers().accepters.contains_key(path)
            || EARLY_PARSED_ATTRIBUTES.contains(&path)
            || SPECIAL_ATTRIBUTES.contains(&path)
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
