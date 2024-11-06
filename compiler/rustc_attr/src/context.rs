use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ops::Deref;
use std::sync::LazyLock;

use rustc_ast::{self as ast, DelimArgs};
use rustc_errors::DiagCtxtHandle;
use rustc_feature::Features;
use rustc_hir::{AttrArgs, AttrItem, AttrPath, Attribute, AttributeKind, HashIgnoredAttrId};
use rustc_session::Session;
use rustc_span::symbol::kw;
use rustc_span::{DUMMY_SP, Span, Symbol, sym};

use crate::attributes::allow_unstable::{AllowConstFnUnstableGroup, AllowInternalUnstableGroup};
use crate::attributes::confusables::ConfusablesGroup;
use crate::attributes::deprecation::DeprecationGroup;
use crate::attributes::repr::ReprGroup;
use crate::attributes::stability::{
    BodyStabilityGroup, ConstStabilityGroup, ConstStabilityIndirectGroup, StabilityGroup,
};
use crate::attributes::transparency::TransparencyGroup;
use crate::attributes::{AttributeFilter, AttributeGroup, Combine, Single};
use crate::parser::{GenericArgParser, GenericMetaItemParser, MetaItemParser};

macro_rules! attribute_groups {
    (
        pub(crate) static $name: ident = [$($names: ty),* $(,)?];
    ) => {
        pub(crate) static $name: LazyLock<(
            BTreeMap<&'static [Symbol], Vec<Box<dyn Fn(&AttributeAcceptContext<'_>, &GenericArgParser<'_, ast::Expr>) + Send + Sync>>>,
            Vec<Box<dyn Send + Sync + Fn(&AttributeGroupContext<'_>) -> Option<(AttributeKind, AttributeFilter)>>>
        )> = LazyLock::new(|| {
            let mut accepts = BTreeMap::<_, Vec<Box<dyn Fn(&AttributeAcceptContext<'_>, &GenericArgParser<'_, ast::Expr>) + Send + Sync>>>::new();
            let mut finalizes = Vec::<Box<dyn Send + Sync + Fn(&AttributeGroupContext<'_>) -> Option<(AttributeKind, AttributeFilter)>>>::new();

            $(
                {
                    thread_local! {
                        static STATE_OBJECT: RefCell<$names> = RefCell::new(<$names>::default());
                    };

                    for (k, v) in <$names>::ATTRIBUTES {
                        accepts.entry(*k).or_default().push(Box::new(|cx, args| {
                            STATE_OBJECT.with_borrow_mut(|s| {
                                v(s, cx, args)
                            })
                        }));
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

attribute_groups!(
    // TODO: rename to the same as in `AttributeDuplicates`
    pub(crate) static ATTRIBUTE_GROUP_MAPPING = [
        // tidy-alphabetical-start
        BodyStabilityGroup,
        ConfusablesGroup,
        ConstStabilityGroup,
        StabilityGroup,
        // tidy-alphabetical-end

        // tidy-alphabetical-start
        Combine<AllowConstFnUnstableGroup>,
        Combine<AllowInternalUnstableGroup>,
        Combine<ReprGroup>,
        // tidy-alphabetical-end

        // tidy-alphabetical-start
        Single<ConstStabilityIndirectGroup>,
        Single<DeprecationGroup>,
        Single<TransparencyGroup>,
        // tidy-alphabetical-end
    ];
);

/// Context created for every attribute that's accepted
///
/// Gives [`AttributeGroup`]s enough information to create errors, for example.
pub(crate) struct AttributeAcceptContext<'a> {
    pub(crate) group_cx: &'a AttributeGroupContext<'a>,
    /// The span of the attribute currently being parsed
    pub(crate) attr_span: Span,
}

impl<'a> Deref for AttributeAcceptContext<'a> {
    type Target = AttributeGroupContext<'a>;

    fn deref(&self) -> &Self::Target {
        &self.group_cx
    }
}

/// Context created for every attribute group that's parsed
///
/// Gives [`AttributeGroup`]s enough information to create errors, for example.
pub(crate) struct AttributeGroupContext<'a> {
    /// The parse context, gives access to the session and the
    /// diagnostics context.
    pub(crate) cx: &'a AttributeParseContext<'a>,
    /// The span of the syntactical component this attribute was applied to
    pub(crate) target_span: Span,
}

impl<'a> Deref for AttributeGroupContext<'a> {
    type Target = AttributeParseContext<'a>;

    fn deref(&self) -> &Self::Target {
        &self.cx
    }
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum OmitDoc {
    Lower,
    Skip,
}

/// Context created once, for example as part of the ast lowering
/// context, through which all attributes can be lowered.
pub struct AttributeParseContext<'sess> {
    tools: Vec<Symbol>,
    sess: &'sess Session,
    features: Option<&'sess Features>,

    /// *only* parse attributes with this symbol.
    ///
    /// Used in cases where we want the lowering infrastructure for
    /// parse just a single attribute.
    parse_only: Option<Symbol>,
}

impl<'sess> AttributeParseContext<'sess> {
    /// This method allows you to parse attributes *before* you have access to features or tools.
    /// One example where this is necessary, is to parse `feature` attributes themselves for
    /// example.
    ///
    /// Try to use this as little as possible. Attributes *should* be lowered during `rustc_ast_lowering`.
    /// Some attributes require access to features to parse, which would crash if you tried to do so
    /// through [`parse_limited`](Self::parse_limited).
    ///
    /// To make sure use is limited, supply a `Symbol` you'd like to parse. Only attributes with
    /// that symbol are picked out of the list of instructions and parsed. Those are returned.
    pub fn parse_limited(
        sess: &'sess Session,
        attrs: &[ast::Attribute],
        sym: Symbol,
        target_span: Span,
    ) -> Option<Attribute> {
        let mut parsed = Self { sess, features: None, tools: Vec::new(), parse_only: Some(sym) }
            .parse_attribute_list(attrs, target_span, OmitDoc::Skip);

        assert!(parsed.len() <= 1);

        parsed.pop()
    }

    pub fn new(sess: &'sess Session, features: &'sess Features, tools: Vec<Symbol>) -> Self {
        Self { sess, features: Some(features), tools, parse_only: None }
    }

    pub(crate) fn sess(&self) -> &'sess Session {
        self.sess
    }

    pub(crate) fn features(&self) -> &'sess Features {
        self.features.expect("features not available at this point in the compiler")
    }

    pub(crate) fn dcx(&self) -> DiagCtxtHandle<'sess> {
        self.sess.dcx()
    }

    /// Parse a list of attributes.
    ///
    /// `target_span` is the span of the thing this list of attributes is applied to,
    /// and when `omit_doc` is set, doc attributes are filtered out.
    pub fn parse_attribute_list<'a>(
        &'a self,
        attrs: &'a [ast::Attribute],
        target_span: Span,
        omit_doc: OmitDoc,
    ) -> Vec<Attribute> {
        let mut attributes = Vec::new();

        let group_cx = AttributeGroupContext { cx: self, target_span };

        for attr in attrs {
            // if we're only looking for a single attribute,
            // skip all the ones we don't care about
            if let Some(expected) = self.parse_only {
                if attr.name_or_empty() != expected {
                    continue;
                }
            }

            // sometimes, for example for `#![doc = include_str!("readme.md")]`,
            // doc still contains a non-literal. You might say, when we're lowering attributes
            // that's expanded right? But no, sometimes, when parsing attributes on macros,
            // we already use the lowering logic and these are still there. So, when `omit_doc`
            // is set we *also* want to ignore these
            if omit_doc == OmitDoc::Skip && attr.name_or_empty() == sym::doc {
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
                        span: attr.span,
                        comment: *symbol,
                    }))
                }
                // // FIXME: make doc attributes go through a proper attribute parser
                // ast::AttrKind::Normal(n) if n.name_or_empty() == sym::doc => {
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
                    let parser = GenericMetaItemParser::from_attr(&n, self.dcx());
                    let (path, args) = parser.deconstruct();
                    let parts = path.segments().map(|i| i.name).collect::<Vec<_>>();

                    if let Some(accepts) = ATTRIBUTE_GROUP_MAPPING.0.get(parts.as_slice()) {
                        for f in accepts {
                            let cx = AttributeAcceptContext {
                                group_cx: &group_cx,
                                attr_span: attr.span,
                            };

                            f(&cx, &args)
                        }
                    } else {
                        // if we're here, we must be compiling a tool attribute... Or someone forgot to
                        // parse their fancy new attribute. Let's warn them in any case. If you are that
                        // person, and you really your attribute should remain unparsed, carefully read the
                        // documentation in this module and if you still think so you can add an exception
                        // to this assertion.

                        const FIXME_TEMPORARY_ATTR_ALLOWLIST: &[Symbol] = &[sym::cfg];
                        assert!(
                            self.tools.contains(&parts[0]) || true,
                            // || FIXME_TEMPORARY_ATTR_ALLOWLIST.contains(&parts[0]),
                            "attribute {path} wasn't parsed and isn't a know tool attribute",
                        );

                        attributes.push(Attribute::Unparsed(Box::new(AttrItem {
                            path: AttrPath::from_ast(&n.item.path),
                            args: self.lower_attr_args(&n.item.args),
                            id: HashIgnoredAttrId { attr_id: attr.id },
                            style: attr.style,
                            span: attr.span,
                        })));
                    }
                }
            }
        }

        let mut parsed_attributes = Vec::new();
        for f in &ATTRIBUTE_GROUP_MAPPING.1 {
            let Some((attr, _filter)) = f(&group_cx) else {
                continue;
            };

            // TODO: ignore filters?
            // todo!("evaluate filters");

            parsed_attributes.push(Attribute::Parsed(attr));
        }

        attributes.extend(parsed_attributes);

        attributes
    }

    fn lower_attr_args(&self, args: &ast::AttrArgs) -> AttrArgs {
        match args {
            ast::AttrArgs::Empty => AttrArgs::Empty,
            ast::AttrArgs::Delimited(args) => AttrArgs::Delimited(DelimArgs {
                dspan: args.dspan,
                delim: args.delim,
                tokens: args.tokens.flattened(),
            }),
            // This is an inert key-value attribute - it will never be visible to macros
            // after it gets lowered to HIR. Therefore, we can extract literals to handle
            // nonterminals in `#[doc]` (e.g. `#[doc = $e]`).
            ast::AttrArgs::Eq { eq_span, expr } => {
                // In valid code the value always ends up as a single literal. Otherwise, a dummy
                // literal suffices because the error is handled elsewhere.
                let lit = if let ast::ExprKind::Lit(token_lit) = expr.kind
                    && let Ok(lit) = ast::MetaItemLit::from_token_lit(token_lit, expr.span)
                {
                    lit
                } else {
                    let guar = self.dcx().has_errors().unwrap();
                    ast::MetaItemLit {
                        symbol: kw::Empty,
                        suffix: None,
                        kind: ast::LitKind::Err(guar),
                        span: DUMMY_SP,
                    }
                };
                AttrArgs::Eq { eq_span: *eq_span, expr: lit }
            }
        }
    }
}
