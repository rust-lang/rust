//! Compiled declarative macro expanders (`macro_rules!` and `macro`)

use std::ops::ControlFlow;

use base_db::{Crate, SourceDatabase};
use span::{Edition, Span, SyntaxContext};
use stdx::TupleExt;
use syntax::{
    AstNode,
    ast::{self, HasAttrs},
};
use syntax_bridge::DocCommentDesugarMode;

use crate::{
    AstId, ExpandError, ExpandErrorKind, ExpandResult, HirFileId, Lookup, MacroCallId,
    MacroCallStyle,
    attrs::{AstKeyValueMetaExt, AstPathExt, expand_cfg_attr},
    hygiene::{Transparency, apply_mark},
    tt,
};

/// Old-style `macro_rules` or the new macros 2.0
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DeclarativeMacroExpander {
    pub mac: mbe::DeclarativeMacro,
    pub transparency: Transparency,
    edition: Edition,
}

impl DeclarativeMacroExpander {
    pub fn expand(
        &self,
        db: &dyn SourceDatabase,
        tt: &tt::TopSubtree,
        call_id: MacroCallId,
        span: Span,
    ) -> ExpandResult<(tt::TopSubtree, Option<u32>)> {
        let loc = call_id.loc(db);
        match self.mac.err() {
            Some(_) => ExpandResult::new(
                (tt::TopSubtree::empty(tt::DelimSpan { open: span, close: span }), None),
                ExpandError::new(span, ExpandErrorKind::MacroDefinition),
            ),
            None => self
                .mac
                .expand(
                    db,
                    tt,
                    |s| {
                        s.ctx =
                            apply_mark(db, s.ctx, call_id.into(), self.transparency, self.edition)
                    },
                    loc.kind.call_style(),
                    span,
                )
                .map_err(Into::into),
        }
    }

    pub fn expand_unhygienic(
        &self,
        db: &dyn SourceDatabase,
        tt: &tt::TopSubtree,
        call_style: MacroCallStyle,
        call_site: Span,
    ) -> ExpandResult<tt::TopSubtree> {
        match self.mac.err() {
            Some(_) => ExpandResult::new(
                tt::TopSubtree::empty(tt::DelimSpan { open: call_site, close: call_site }),
                ExpandError::new(call_site, ExpandErrorKind::MacroDefinition),
            ),
            None => self
                .mac
                .expand(db, tt, |_| (), call_style, call_site)
                .map(TupleExt::head)
                .map_err(Into::into),
        }
    }
}

#[salsa::tracked]
impl AstId<ast::Macro> {
    /// Fetches (and compiles) the expander of this decl macro.
    #[salsa::tracked(returns(ref))]
    pub fn decl_macro_expander(
        self,
        db: &dyn SourceDatabase,
        def_crate: Crate,
    ) -> DeclarativeMacroExpander {
        let id = self;
        let (root, map) = id.file_id.parse_with_map(db);

        let root = root.syntax_node();

        let transparency = |node: ast::AnyHasAttrs| {
            let mut cfg_options = None;
            expand_cfg_attr(
                node.attrs(),
                || cfg_options.get_or_insert_with(|| def_crate.cfg_options(db)),
                |attr, _| {
                    if let ast::Meta::KeyValueMeta(attr) = attr
                        && attr.path().is1("rustc_macro_transparency")
                        && let Some(value) = attr.value_string()
                    {
                        match &*value {
                            "transparent" => ControlFlow::Break(Transparency::Transparent),
                            // "semitransparent" is for old rustc versions.
                            "semiopaque" | "semitransparent" => {
                                ControlFlow::Break(Transparency::SemiOpaque)
                            }
                            "opaque" => ControlFlow::Break(Transparency::Opaque),
                            _ => ControlFlow::Continue(()),
                        }
                    } else {
                        ControlFlow::Continue(())
                    }
                },
            )
        };
        let ctx_edition = |ctx: SyntaxContext| {
            if ctx.is_root() {
                def_crate.data(db).edition
            } else {
                // UNWRAP-SAFETY: Only the root context has no outer expansion
                let krate = crate::MacroCallId::from(ctx.outer_expn(db).unwrap()).loc(db).def.krate;
                krate.data(db).edition
            }
        };
        let (mac, transparency) = match id.to_ptr(db).to_node(&root) {
            ast::Macro::MacroRules(macro_rules) => (
                match macro_rules.token_tree() {
                    Some(arg) => {
                        let tt = syntax_bridge::syntax_node_to_token_tree(
                            arg.syntax(),
                            map,
                            map.span_for_range(
                                macro_rules.macro_rules_token().unwrap().text_range(),
                            ),
                            DocCommentDesugarMode::Mbe,
                        );

                        mbe::DeclarativeMacro::parse_macro_rules(&tt, ctx_edition)
                    }
                    None => mbe::DeclarativeMacro::from_err(mbe::ParseError::Expected(
                        "expected a token tree".into(),
                    )),
                },
                transparency(ast::AnyHasAttrs::from(macro_rules))
                    .unwrap_or(Transparency::SemiOpaque),
            ),
            ast::Macro::MacroDef(macro_def) => (
                match macro_def.body() {
                    Some(body) => {
                        let span =
                            map.span_for_range(macro_def.macro_token().unwrap().text_range());
                        let args = macro_def.args().map(|args| {
                            syntax_bridge::syntax_node_to_token_tree(
                                args.syntax(),
                                map,
                                span,
                                DocCommentDesugarMode::Mbe,
                            )
                        });
                        let body = syntax_bridge::syntax_node_to_token_tree(
                            body.syntax(),
                            map,
                            span,
                            DocCommentDesugarMode::Mbe,
                        );

                        mbe::DeclarativeMacro::parse_macro2(args.as_ref(), &body, ctx_edition)
                    }
                    None => mbe::DeclarativeMacro::from_err(mbe::ParseError::Expected(
                        "expected a token tree".into(),
                    )),
                },
                transparency(macro_def.into()).unwrap_or(Transparency::Opaque),
            ),
        };
        let edition = ctx_edition(match id.file_id {
            HirFileId::MacroFile(macro_file) => macro_file.lookup(db).ctxt,
            HirFileId::FileId(file) => SyntaxContext::root(file.edition(db)),
        });
        DeclarativeMacroExpander { mac, transparency, edition }
    }
}
