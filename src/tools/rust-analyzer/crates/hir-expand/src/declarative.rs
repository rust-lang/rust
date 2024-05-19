//! Compiled declarative macro expanders (`macro_rules!`` and `macro`)
use std::sync::OnceLock;

use base_db::{CrateId, VersionReq};
use mbe::DocCommentDesugarMode;
use span::{Edition, MacroCallId, Span, SyntaxContextId};
use stdx::TupleExt;
use syntax::{ast, AstNode};
use triomphe::Arc;

use crate::{
    attrs::RawAttrs,
    db::ExpandDatabase,
    hygiene::{apply_mark, Transparency},
    tt, AstId, ExpandError, ExpandResult, Lookup,
};

/// Old-style `macro_rules` or the new macros 2.0
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DeclarativeMacroExpander {
    pub mac: mbe::DeclarativeMacro,
    pub transparency: Transparency,
}

// FIXME: Remove this once we drop support for 1.76
static REQUIREMENT: OnceLock<VersionReq> = OnceLock::new();

impl DeclarativeMacroExpander {
    pub fn expand(
        &self,
        db: &dyn ExpandDatabase,
        tt: tt::Subtree,
        call_id: MacroCallId,
        span: Span,
    ) -> ExpandResult<(tt::Subtree, Option<u32>)> {
        let loc = db.lookup_intern_macro_call(call_id);
        let toolchain = db.toolchain(loc.def.krate);
        let new_meta_vars = toolchain.as_ref().map_or(false, |version| {
            REQUIREMENT.get_or_init(|| VersionReq::parse(">=1.76").unwrap()).matches(
                &base_db::Version {
                    pre: base_db::Prerelease::EMPTY,
                    build: base_db::BuildMetadata::EMPTY,
                    major: version.major,
                    minor: version.minor,
                    patch: version.patch,
                },
            )
        });
        match self.mac.err() {
            Some(_) => ExpandResult::new(
                (tt::Subtree::empty(tt::DelimSpan { open: span, close: span }), None),
                ExpandError::MacroDefinition,
            ),
            None => self
                .mac
                .expand(
                    &tt,
                    |s| s.ctx = apply_mark(db, s.ctx, call_id, self.transparency),
                    new_meta_vars,
                    span,
                    loc.def.edition,
                )
                .map_err(Into::into),
        }
    }

    pub fn expand_unhygienic(
        &self,
        db: &dyn ExpandDatabase,
        tt: tt::Subtree,
        krate: CrateId,
        call_site: Span,
        def_site_edition: Edition,
    ) -> ExpandResult<tt::Subtree> {
        let toolchain = db.toolchain(krate);
        let new_meta_vars = toolchain.as_ref().map_or(false, |version| {
            REQUIREMENT.get_or_init(|| VersionReq::parse(">=1.76").unwrap()).matches(
                &base_db::Version {
                    pre: base_db::Prerelease::EMPTY,
                    build: base_db::BuildMetadata::EMPTY,
                    major: version.major,
                    minor: version.minor,
                    patch: version.patch,
                },
            )
        });
        match self.mac.err() {
            Some(_) => ExpandResult::new(
                tt::Subtree::empty(tt::DelimSpan { open: call_site, close: call_site }),
                ExpandError::MacroDefinition,
            ),
            None => self
                .mac
                .expand(&tt, |_| (), new_meta_vars, call_site, def_site_edition)
                .map(TupleExt::head)
                .map_err(Into::into),
        }
    }

    pub(crate) fn expander(
        db: &dyn ExpandDatabase,
        def_crate: CrateId,
        id: AstId<ast::Macro>,
    ) -> Arc<DeclarativeMacroExpander> {
        let (root, map) = crate::db::parse_with_map(db, id.file_id);
        let root = root.syntax_node();

        let transparency = |node| {
            // ... would be nice to have the item tree here
            let attrs = RawAttrs::new(db, node, map.as_ref()).filter(db, def_crate);
            match &*attrs
                .iter()
                .find(|it| {
                    it.path.as_ident().and_then(|it| it.as_str())
                        == Some("rustc_macro_transparency")
                })?
                .token_tree_value()?
                .token_trees
            {
                [tt::TokenTree::Leaf(tt::Leaf::Ident(i)), ..] => match &*i.text {
                    "transparent" => Some(Transparency::Transparent),
                    "semitransparent" => Some(Transparency::SemiTransparent),
                    "opaque" => Some(Transparency::Opaque),
                    _ => None,
                },
                _ => None,
            }
        };
        let toolchain = db.toolchain(def_crate);
        let new_meta_vars = toolchain.as_ref().map_or(false, |version| {
            REQUIREMENT.get_or_init(|| VersionReq::parse(">=1.76").unwrap()).matches(
                &base_db::Version {
                    pre: base_db::Prerelease::EMPTY,
                    build: base_db::BuildMetadata::EMPTY,
                    major: version.major,
                    minor: version.minor,
                    patch: version.patch,
                },
            )
        });

        let edition = |ctx: SyntaxContextId| {
            let crate_graph = db.crate_graph();
            if ctx.is_root() {
                crate_graph[def_crate].edition
            } else {
                let data = db.lookup_intern_syntax_context(ctx);
                // UNWRAP-SAFETY: Only the root context has no outer expansion
                crate_graph[data.outer_expn.unwrap().lookup(db).def.krate].edition
            }
        };
        let (mac, transparency) = match id.to_ptr(db).to_node(&root) {
            ast::Macro::MacroRules(macro_rules) => (
                match macro_rules.token_tree() {
                    Some(arg) => {
                        let tt = mbe::syntax_node_to_token_tree(
                            arg.syntax(),
                            map.as_ref(),
                            map.span_for_range(
                                macro_rules.macro_rules_token().unwrap().text_range(),
                            ),
                            DocCommentDesugarMode::Mbe,
                        );

                        mbe::DeclarativeMacro::parse_macro_rules(&tt, edition, new_meta_vars)
                    }
                    None => mbe::DeclarativeMacro::from_err(mbe::ParseError::Expected(
                        "expected a token tree".into(),
                    )),
                },
                transparency(&macro_rules).unwrap_or(Transparency::SemiTransparent),
            ),
            ast::Macro::MacroDef(macro_def) => (
                match macro_def.body() {
                    Some(arg) => {
                        let tt = mbe::syntax_node_to_token_tree(
                            arg.syntax(),
                            map.as_ref(),
                            map.span_for_range(macro_def.macro_token().unwrap().text_range()),
                            DocCommentDesugarMode::Mbe,
                        );

                        mbe::DeclarativeMacro::parse_macro2(&tt, edition, new_meta_vars)
                    }
                    None => mbe::DeclarativeMacro::from_err(mbe::ParseError::Expected(
                        "expected a token tree".into(),
                    )),
                },
                transparency(&macro_def).unwrap_or(Transparency::Opaque),
            ),
        };
        Arc::new(DeclarativeMacroExpander { mac, transparency })
    }
}
