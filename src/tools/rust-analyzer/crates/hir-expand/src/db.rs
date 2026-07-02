//! Defines database & queries for macro expansion.

use base_db::{Crate, SourceDatabase};
use span::{AstIdMap, Span};
use syntax::{AstNode, Parse, SyntaxNode, SyntaxToken, ast};
use syntax_bridge::{DocCommentDesugarMode, syntax_node_to_token_tree};

use crate::{
    AstId, BuiltinAttrExpander, BuiltinDeriveExpander, BuiltinFnLikeExpander, EagerExpander,
    EditionedFileId, FileRange, HirFileId, MacroCallId, MacroCallKind, MacroDefId, MacroDefKind,
    builtin::pseudo_derive_attr_expansion,
    cfg_process::attr_macro_input_to_token_tree,
    declarative::DeclarativeMacroExpander,
    fixup::{self, SyntaxFixupUndoInfo},
    hygiene::{span_with_call_site_ctxt, span_with_def_site_ctxt, span_with_mixed_site_ctxt},
    proc_macro::CustomProcMacroExpander,
    span_map::{ExpansionSpanMap, RealSpanMap, SpanMap},
    tt,
};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TokenExpander<'db> {
    /// Old-style `macro_rules` or the new macros 2.0
    DeclarativeMacro(&'db DeclarativeMacroExpander),
    /// Stuff like `line!` and `file!`.
    BuiltIn(BuiltinFnLikeExpander),
    /// Built-in eagerly expanded fn-like macros (`include!`, `concat!`, etc.)
    BuiltInEager(EagerExpander),
    /// `global_allocator` and such.
    BuiltInAttr(BuiltinAttrExpander),
    /// `derive(Copy)` and such.
    BuiltInDerive(BuiltinDeriveExpander),
    UnimplementedBuiltIn,
    /// The thing we love the most here in rust-analyzer -- procedural macros.
    ProcMacro(CustomProcMacroExpander),
}

#[query_group::query_group]
pub trait ExpandDatabase: SourceDatabase {
    #[salsa::invoke(ast_id_map)]
    #[salsa::transparent]
    fn ast_id_map(&self, file_id: HirFileId) -> &AstIdMap;

    #[salsa::transparent]
    fn resolve_span(&self, span: Span) -> FileRange;

    #[salsa::transparent]
    fn parse_or_expand(&self, file_id: HirFileId) -> SyntaxNode;

    #[salsa::transparent]
    #[salsa::invoke(SpanMap::new)]
    fn span_map(&self, file_id: HirFileId) -> SpanMap<'_>;

    #[salsa::transparent]
    #[salsa::invoke(crate::span_map::expansion_span_map)]
    fn expansion_span_map(&self, file_id: MacroCallId) -> &ExpansionSpanMap;
    #[salsa::invoke(crate::span_map::real_span_map)]
    #[salsa::transparent]
    fn real_span_map(&self, file_id: EditionedFileId) -> &RealSpanMap;

    /// Fetches the expander for this macro.
    #[salsa::transparent]
    #[salsa::invoke(TokenExpander::macro_expander)]
    fn macro_expander(&self, id: MacroDefId) -> TokenExpander<'_>;

    /// Fetches (and compiles) the expander of this decl macro.
    #[salsa::invoke(DeclarativeMacroExpander::expander)]
    #[salsa::transparent]
    fn decl_macro_expander(
        &self,
        def_crate: Crate,
        id: AstId<ast::Macro>,
    ) -> &DeclarativeMacroExpander;
}

fn resolve_span(db: &dyn ExpandDatabase, Span { range, anchor, ctx: _ }: Span) -> FileRange {
    let file_id = EditionedFileId::from_span_file_id(db, anchor.file_id);
    let anchor_offset =
        db.ast_id_map(file_id.into()).get_erased(anchor.ast_id).text_range().start();
    FileRange { file_id, range: range + anchor_offset }
}

/// This expands the given macro call, but with different arguments. This is
/// used for completion, where we want to see what 'would happen' if we insert a
/// token. The `token_to_map` mapped down into the expansion, with the mapped
/// token(s) returned with their priority.
pub fn expand_speculative(
    db: &dyn ExpandDatabase,
    actual_macro_call: MacroCallId,
    speculative_args: &SyntaxNode,
    token_to_map: SyntaxToken,
) -> Option<(SyntaxNode, Vec<(SyntaxToken, u8)>)> {
    let loc = actual_macro_call.loc(db);
    let (_, _, span) = *actual_macro_call.macro_arg_considering_derives(db, &loc.kind);

    let span_map = RealSpanMap::absolute(span.anchor.file_id);
    let span_map = SpanMap::RealSpanMap(&span_map);

    // Build the subtree and token mapping for the speculative args
    let (mut tt, undo_info) = match &loc.kind {
        MacroCallKind::FnLike { .. } => (
            syntax_bridge::syntax_node_to_token_tree(
                speculative_args,
                span_map,
                span,
                if loc.def.is_proc_macro() {
                    DocCommentDesugarMode::ProcMacro
                } else {
                    DocCommentDesugarMode::Mbe
                },
            ),
            SyntaxFixupUndoInfo::NONE,
        ),
        MacroCallKind::Attr { .. } if loc.def.is_attribute_derive() => (
            syntax_bridge::syntax_node_to_token_tree(
                speculative_args,
                span_map,
                span,
                DocCommentDesugarMode::ProcMacro,
            ),
            SyntaxFixupUndoInfo::NONE,
        ),
        MacroCallKind::Derive { derive_macro_id, .. } => {
            let MacroCallKind::Attr { censored_attr_ids: attr_ids, .. } =
                &derive_macro_id.loc(db).kind
            else {
                unreachable!("`derive_macro_id` should be `MacroCallKind::Attr`");
            };
            attr_macro_input_to_token_tree(
                db,
                speculative_args,
                span_map,
                span,
                true,
                attr_ids,
                loc.krate,
            )
        }
        MacroCallKind::Attr { censored_attr_ids: attr_ids, .. } => attr_macro_input_to_token_tree(
            db,
            speculative_args,
            span_map,
            span,
            false,
            attr_ids,
            loc.krate,
        ),
    };

    let attr_arg = match &loc.kind {
        MacroCallKind::Attr { censored_attr_ids: attr_ids, .. } => {
            if loc.def.is_attribute_derive() {
                // for pseudo-derive expansion we actually pass the attribute itself only
                ast::Attr::cast(speculative_args.clone())
                    .and_then(|attr| {
                        if let ast::Meta::TokenTreeMeta(meta) = attr.meta()? {
                            meta.token_tree()
                        } else {
                            None
                        }
                    })
                    .map(|token_tree| {
                        let mut tree = syntax_node_to_token_tree(
                            token_tree.syntax(),
                            span_map,
                            span,
                            DocCommentDesugarMode::ProcMacro,
                        );
                        tree.set_top_subtree_delimiter_kind(tt::DelimiterKind::Invisible);
                        tree.set_top_subtree_delimiter_span(tt::DelimSpan::from_single(span));
                        tree
                    })
            } else {
                // Attributes may have an input token tree, build the subtree and map for this as well
                // then try finding a token id for our token if it is inside this input subtree.
                let item = ast::Item::cast(speculative_args.clone())?;
                let (_, meta) =
                    attr_ids.invoc_attr().find_attr_range_with_source_opt(db, loc.krate, &item)?;
                if let ast::Meta::TokenTreeMeta(meta) = meta
                    && let Some(tt) = meta.token_tree()
                {
                    let mut attr_arg = syntax_bridge::syntax_node_to_token_tree(
                        tt.syntax(),
                        span_map,
                        span,
                        DocCommentDesugarMode::ProcMacro,
                    );
                    attr_arg.set_top_subtree_delimiter_kind(tt::DelimiterKind::Invisible);
                    Some(attr_arg)
                } else {
                    None
                }
            }
        }
        _ => None,
    };

    // Do the actual expansion, we need to directly expand the proc macro due to the attribute args
    // Otherwise the expand query will fetch the non speculative attribute args and pass those instead.
    let mut speculative_expansion = match loc.def.kind {
        MacroDefKind::ProcMacro(ast, expander, _) => {
            let span = crate::proc_macro_span(db, ast);
            tt.set_top_subtree_delimiter_kind(tt::DelimiterKind::Invisible);
            tt.set_top_subtree_delimiter_span(tt::DelimSpan::from_single(span));
            expander.expand(
                db,
                loc.def.krate,
                loc.krate,
                &tt,
                attr_arg.as_ref(),
                span_with_def_site_ctxt(db, span, actual_macro_call.into(), loc.def.edition),
                span_with_call_site_ctxt(db, span, actual_macro_call.into(), loc.def.edition),
                span_with_mixed_site_ctxt(db, span, actual_macro_call.into(), loc.def.edition),
            )
        }
        MacroDefKind::BuiltInAttr(_, it) if it.is_derive() => {
            pseudo_derive_attr_expansion(&tt, attr_arg.as_ref()?, span)
        }
        MacroDefKind::Declarative(it, _) => db
            .decl_macro_expander(loc.krate, it)
            .expand_unhygienic(db, &tt, loc.kind.call_style(), span),
        MacroDefKind::BuiltIn(_, it) => {
            it.expand(db, actual_macro_call, &tt, span).map_err(Into::into)
        }
        MacroDefKind::BuiltInDerive(_, it) => {
            it.expand(db, actual_macro_call, &tt, span).map_err(Into::into)
        }
        MacroDefKind::BuiltInEager(_, it) => {
            it.expand(db, actual_macro_call, &tt, span).map_err(Into::into)
        }
        MacroDefKind::BuiltInAttr(_, it) => it.expand(db, actual_macro_call, &tt, span),
        MacroDefKind::UnimplementedBuiltIn(_) => crate::expand_unimplemented_builtin_macro(span),
    };

    let expand_to = loc.expand_to();

    fixup::reverse_fixups(&mut speculative_expansion.value, &undo_info);
    let (node, rev_tmap) =
        crate::token_tree_to_syntax_node(db, &speculative_expansion.value, expand_to);

    let syntax_node = node.syntax_node();
    let token = rev_tmap
        .ranges_with_span(span_map.span_for_range(token_to_map.text_range()))
        .filter_map(|(range, ctx)| syntax_node.covering_element(range).into_token().zip(Some(ctx)))
        .map(|(t, ctx)| {
            // prefer tokens of the same kind and text, as well as non opaque marked ones
            // Note the inversion of the score here, as we want to prefer the first token in case
            // of all tokens having the same score
            let ranking = ctx.is_opaque(db) as u8
                + 2 * (t.kind() != token_to_map.kind()) as u8
                + 4 * ((t.text() != token_to_map.text()) as u8);
            (t, ranking)
        })
        .collect();
    Some((node.syntax_node(), token))
}

#[salsa::tracked(lru = 1024, returns(ref))]
fn ast_id_map(db: &dyn ExpandDatabase, file_id: HirFileId) -> AstIdMap {
    AstIdMap::from_source(&db.parse_or_expand(file_id))
}

/// Main public API -- parses a hir file, not caring whether it's a real
/// file or a macro expansion.
fn parse_or_expand(db: &dyn ExpandDatabase, file_id: HirFileId) -> SyntaxNode {
    match file_id {
        HirFileId::FileId(file_id) => file_id.parse(db).syntax_node(),
        HirFileId::MacroFile(macro_file) => {
            macro_file.parse_macro_expansion(db).value.0.syntax_node()
        }
    }
}

pub(crate) fn parse_with_map(
    db: &dyn ExpandDatabase,
    file_id: HirFileId,
) -> (Parse<SyntaxNode>, SpanMap<'_>) {
    match file_id {
        HirFileId::FileId(file_id) => {
            (file_id.parse(db).to_syntax(), SpanMap::RealSpanMap(db.real_span_map(file_id)))
        }
        HirFileId::MacroFile(macro_file) => {
            let (parse, map) = &macro_file.parse_macro_expansion(db).value;
            (parse.clone(), SpanMap::ExpansionSpanMap(map))
        }
    }
}

impl<'db> TokenExpander<'db> {
    fn macro_expander(db: &'db dyn ExpandDatabase, id: MacroDefId) -> TokenExpander<'db> {
        match id.kind {
            MacroDefKind::Declarative(ast_id, _) => {
                TokenExpander::DeclarativeMacro(db.decl_macro_expander(id.krate, ast_id))
            }
            MacroDefKind::BuiltIn(_, expander) => TokenExpander::BuiltIn(expander),
            MacroDefKind::BuiltInAttr(_, expander) => TokenExpander::BuiltInAttr(expander),
            MacroDefKind::BuiltInDerive(_, expander) => TokenExpander::BuiltInDerive(expander),
            MacroDefKind::BuiltInEager(_, expander) => TokenExpander::BuiltInEager(expander),
            MacroDefKind::ProcMacro(_, expander, _) => TokenExpander::ProcMacro(expander),
            MacroDefKind::UnimplementedBuiltIn(_) => TokenExpander::UnimplementedBuiltIn,
        }
    }
}
