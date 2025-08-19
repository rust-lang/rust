//! Compiled declarative macro expanders (`macro_rules!` and `macro`)

use base_db::Crate;
use intern::sym;
use span::{Edition, Span, SyntaxContext};
use stdx::TupleExt;
use syntax::{AstNode, ast};
use syntax_bridge::DocCommentDesugarMode;
use triomphe::Arc;

use crate::{
    AstId, ExpandError, ExpandErrorKind, ExpandResult, HirFileId, Lookup, MacroCallId,
    attrs::RawAttrs,
    db::ExpandDatabase,
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
        db: &dyn ExpandDatabase,
        tt: tt::TopSubtree,
        call_id: MacroCallId,
        span: Span,
    ) -> ExpandResult<(tt::TopSubtree, Option<u32>)> {
        let loc = db.lookup_intern_macro_call(call_id);
        match self.mac.err() {
            Some(_) => ExpandResult::new(
                (tt::TopSubtree::empty(tt::DelimSpan { open: span, close: span }), None),
                ExpandError::new(span, ExpandErrorKind::MacroDefinition),
            ),
            None => self
                .mac
                .expand(
                    &tt,
                    |s| {
                        s.ctx =
                            apply_mark(db, s.ctx, call_id.into(), self.transparency, self.edition)
                    },
                    span,
                    loc.def.edition,
                )
                .map_err(Into::into),
        }
    }

    pub fn expand_unhygienic(
        &self,
        tt: tt::TopSubtree,
        call_site: Span,
        def_site_edition: Edition,
    ) -> ExpandResult<tt::TopSubtree> {
        match self.mac.err() {
            Some(_) => ExpandResult::new(
                tt::TopSubtree::empty(tt::DelimSpan { open: call_site, close: call_site }),
                ExpandError::new(call_site, ExpandErrorKind::MacroDefinition),
            ),
            None => self
                .mac
                .expand(&tt, |_| (), call_site, def_site_edition)
                .map(TupleExt::head)
                .map_err(Into::into),
        }
    }

    pub(crate) fn expander(
        db: &dyn ExpandDatabase,
        def_crate: Crate,
        id: AstId<ast::Macro>,
    ) -> Arc<DeclarativeMacroExpander> {
        let (root, map) = crate::db::parse_with_map(db, id.file_id);
        let root = root.syntax_node();

        let transparency = |node| {
            // ... would be nice to have the item tree here
            let attrs = RawAttrs::new_expanded(db, node, map.as_ref(), def_crate.cfg_options(db));
            match attrs
                .iter()
                .find(|it| {
                    it.path
                        .as_ident()
                        .map(|it| *it == sym::rustc_macro_transparency)
                        .unwrap_or(false)
                })?
                .token_tree_value()?
                .token_trees()
                .flat_tokens()
            {
                [tt::TokenTree::Leaf(tt::Leaf::Ident(i)), ..] => match &i.sym {
                    s if *s == sym::transparent => Some(Transparency::Transparent),
                    s if *s == sym::semitransparent => Some(Transparency::SemiTransparent),
                    s if *s == sym::opaque => Some(Transparency::Opaque),
                    _ => None,
                },
                _ => None,
            }
        };
        let ctx_edition = |ctx: SyntaxContext| {
            if ctx.is_root() {
                def_crate.data(db).edition
            } else {
                // UNWRAP-SAFETY: Only the root context has no outer expansion
                let krate =
                    db.lookup_intern_macro_call(ctx.outer_expn(db).unwrap().into()).def.krate;
                krate.data(db).edition
            }
        };
        let (mac, transparency) = match id.to_ptr(db).to_node(&root) {
            ast::Macro::MacroRules(macro_rules) => (
                match macro_rules.token_tree() {
                    Some(arg) => {
                        let tt = syntax_bridge::syntax_node_to_token_tree(
                            arg.syntax(),
                            map.as_ref(),
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
                transparency(&macro_rules).unwrap_or(Transparency::SemiTransparent),
            ),
            ast::Macro::MacroDef(macro_def) => (
                match macro_def.body() {
                    Some(body) => {
                        let span =
                            map.span_for_range(macro_def.macro_token().unwrap().text_range());
                        let args = macro_def.args().map(|args| {
                            syntax_bridge::syntax_node_to_token_tree(
                                args.syntax(),
                                map.as_ref(),
                                span,
                                DocCommentDesugarMode::Mbe,
                            )
                        });
                        let body = syntax_bridge::syntax_node_to_token_tree(
                            body.syntax(),
                            map.as_ref(),
                            span,
                            DocCommentDesugarMode::Mbe,
                        );

                        mbe::DeclarativeMacro::parse_macro2(args.as_ref(), &body, ctx_edition)
                    }
                    None => mbe::DeclarativeMacro::from_err(mbe::ParseError::Expected(
                        "expected a token tree".into(),
                    )),
                },
                transparency(&macro_def).unwrap_or(Transparency::Opaque),
            ),
        };
        let edition = ctx_edition(match id.file_id {
            HirFileId::MacroFile(macro_file) => macro_file.lookup(db).ctxt,
            HirFileId::FileId(file) => SyntaxContext::root(file.edition(db)),
        });
        Arc::new(DeclarativeMacroExpander { mac, transparency, edition })
    }
}
