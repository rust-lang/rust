//! Defines database & queries for macro expansion.

use base_db::{salsa, CrateId, FileId, SourceDatabase};
use either::Either;
use limit::Limit;
use mbe::syntax_node_to_token_tree;
use rustc_hash::FxHashSet;
use span::{AstIdMap, Span, SyntaxContextData, SyntaxContextId};
use syntax::{ast, AstNode, Parse, SyntaxElement, SyntaxError, SyntaxNode, SyntaxToken, T};
use triomphe::Arc;

use crate::{
    attrs::{collect_attrs, AttrId},
    builtin_attr_macro::pseudo_derive_attr_expansion,
    builtin_fn_macro::EagerExpander,
    cfg_process,
    declarative::DeclarativeMacroExpander,
    fixup::{self, SyntaxFixupUndoInfo},
    hygiene::{span_with_call_site_ctxt, span_with_def_site_ctxt, span_with_mixed_site_ctxt},
    proc_macro::ProcMacros,
    span_map::{RealSpanMap, SpanMap, SpanMapRef},
    tt, AstId, BuiltinAttrExpander, BuiltinDeriveExpander, BuiltinFnLikeExpander,
    CustomProcMacroExpander, EagerCallInfo, ExpandError, ExpandResult, ExpandTo, ExpansionSpanMap,
    HirFileId, HirFileIdRepr, MacroCallId, MacroCallKind, MacroCallLoc, MacroDefId, MacroDefKind,
    MacroFileId,
};

/// Total limit on the number of tokens produced by any macro invocation.
///
/// If an invocation produces more tokens than this limit, it will not be stored in the database and
/// an error will be emitted.
///
/// Actual max for `analysis-stats .` at some point: 30672.
static TOKEN_LIMIT: Limit = Limit::new(1_048_576);

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum TokenExpander {
    /// Old-style `macro_rules` or the new macros 2.0
    DeclarativeMacro(Arc<DeclarativeMacroExpander>),
    /// Stuff like `line!` and `file!`.
    BuiltIn(BuiltinFnLikeExpander),
    /// Built-in eagerly expanded fn-like macros (`include!`, `concat!`, etc.)
    BuiltInEager(EagerExpander),
    /// `global_allocator` and such.
    BuiltInAttr(BuiltinAttrExpander),
    /// `derive(Copy)` and such.
    BuiltInDerive(BuiltinDeriveExpander),
    /// The thing we love the most here in rust-analyzer -- procedural macros.
    ProcMacro(CustomProcMacroExpander),
}

#[salsa::query_group(ExpandDatabaseStorage)]
pub trait ExpandDatabase: SourceDatabase {
    /// The proc macros.
    #[salsa::input]
    fn proc_macros(&self) -> Arc<ProcMacros>;

    fn ast_id_map(&self, file_id: HirFileId) -> Arc<AstIdMap>;

    /// Main public API -- parses a hir file, not caring whether it's a real
    /// file or a macro expansion.
    #[salsa::transparent]
    fn parse_or_expand(&self, file_id: HirFileId) -> SyntaxNode;
    #[salsa::transparent]
    fn parse_or_expand_with_err(&self, file_id: HirFileId) -> ExpandResult<Parse<SyntaxNode>>;
    /// Implementation for the macro case.
    // This query is LRU cached
    fn parse_macro_expansion(
        &self,
        macro_file: MacroFileId,
    ) -> ExpandResult<(Parse<SyntaxNode>, Arc<ExpansionSpanMap>)>;
    #[salsa::transparent]
    #[salsa::invoke(SpanMap::new)]
    fn span_map(&self, file_id: HirFileId) -> SpanMap;

    #[salsa::transparent]
    #[salsa::invoke(crate::span_map::expansion_span_map)]
    fn expansion_span_map(&self, file_id: MacroFileId) -> Arc<ExpansionSpanMap>;
    #[salsa::invoke(crate::span_map::real_span_map)]
    fn real_span_map(&self, file_id: FileId) -> Arc<RealSpanMap>;

    /// Macro ids. That's probably the tricksiest bit in rust-analyzer, and the
    /// reason why we use salsa at all.
    ///
    /// We encode macro definitions into ids of macro calls, this what allows us
    /// to be incremental.
    #[salsa::interned]
    fn intern_macro_call(&self, macro_call: MacroCallLoc) -> MacroCallId;
    #[salsa::interned]
    fn intern_syntax_context(&self, ctx: SyntaxContextData) -> SyntaxContextId;

    #[salsa::transparent]
    fn setup_syntax_context_root(&self) -> ();
    #[salsa::transparent]
    #[salsa::invoke(crate::hygiene::dump_syntax_contexts)]
    fn dump_syntax_contexts(&self) -> String;

    /// Lowers syntactic macro call to a token tree representation. That's a firewall
    /// query, only typing in the macro call itself changes the returned
    /// subtree.
    fn macro_arg(&self, id: MacroCallId) -> (Arc<tt::Subtree>, SyntaxFixupUndoInfo, Span);
    /// Fetches the expander for this macro.
    #[salsa::transparent]
    #[salsa::invoke(TokenExpander::macro_expander)]
    fn macro_expander(&self, id: MacroDefId) -> TokenExpander;
    /// Fetches (and compiles) the expander of this decl macro.
    #[salsa::invoke(DeclarativeMacroExpander::expander)]
    fn decl_macro_expander(
        &self,
        def_crate: CrateId,
        id: AstId<ast::Macro>,
    ) -> Arc<DeclarativeMacroExpander>;
    /// Special case of the previous query for procedural macros. We can't LRU
    /// proc macros, since they are not deterministic in general, and
    /// non-determinism breaks salsa in a very, very, very bad way.
    /// @edwin0cheng heroically debugged this once! See #4315 for details
    fn expand_proc_macro(&self, call: MacroCallId) -> ExpandResult<Arc<tt::Subtree>>;
    /// Retrieves the span to be used for a proc-macro expansions spans.
    /// This is a firewall query as it requires parsing the file, which we don't want proc-macros to
    /// directly depend on as that would cause to frequent invalidations, mainly because of the
    /// parse queries being LRU cached. If they weren't the invalidations would only happen if the
    /// user wrote in the file that defines the proc-macro.
    fn proc_macro_span(&self, fun: AstId<ast::Fn>) -> Span;
    /// Firewall query that returns the errors from the `parse_macro_expansion` query.
    fn parse_macro_expansion_error(
        &self,
        macro_call: MacroCallId,
    ) -> ExpandResult<Box<[SyntaxError]>>;
}

/// This expands the given macro call, but with different arguments. This is
/// used for completion, where we want to see what 'would happen' if we insert a
/// token. The `token_to_map` mapped down into the expansion, with the mapped
/// token returned.
pub fn expand_speculative(
    db: &dyn ExpandDatabase,
    actual_macro_call: MacroCallId,
    speculative_args: &SyntaxNode,
    token_to_map: SyntaxToken,
) -> Option<(SyntaxNode, SyntaxToken)> {
    let loc = db.lookup_intern_macro_call(actual_macro_call);

    // FIXME: This BOGUS here is dangerous once the proc-macro server can call back into the database!
    let span_map = RealSpanMap::absolute(FileId::BOGUS);
    let span_map = SpanMapRef::RealSpanMap(&span_map);

    let (_, _, span) = db.macro_arg(actual_macro_call);

    // Build the subtree and token mapping for the speculative args
    let (mut tt, undo_info) = match loc.kind {
        MacroCallKind::FnLike { .. } => (
            mbe::syntax_node_to_token_tree(speculative_args, span_map, span),
            SyntaxFixupUndoInfo::NONE,
        ),
        MacroCallKind::Attr { .. } if loc.def.is_attribute_derive() => (
            mbe::syntax_node_to_token_tree(speculative_args, span_map, span),
            SyntaxFixupUndoInfo::NONE,
        ),
        MacroCallKind::Derive { derive_attr_index: index, .. }
        | MacroCallKind::Attr { invoc_attr_index: index, .. } => {
            let censor = if let MacroCallKind::Derive { .. } = loc.kind {
                censor_derive_input(index, &ast::Adt::cast(speculative_args.clone())?)
            } else {
                attr_source(index, &ast::Item::cast(speculative_args.clone())?)
                    .into_iter()
                    .map(|it| it.syntax().clone().into())
                    .collect()
            };

            let censor_cfg =
                cfg_process::process_cfg_attrs(speculative_args, &loc, db).unwrap_or_default();
            let mut fixups = fixup::fixup_syntax(span_map, speculative_args, span);
            fixups.append.retain(|it, _| match it {
                syntax::NodeOrToken::Token(_) => true,
                it => !censor.contains(it) && !censor_cfg.contains(it),
            });
            fixups.remove.extend(censor);
            fixups.remove.extend(censor_cfg);

            (
                mbe::syntax_node_to_token_tree_modified(
                    speculative_args,
                    span_map,
                    fixups.append,
                    fixups.remove,
                    span,
                ),
                fixups.undo_info,
            )
        }
    };

    let attr_arg = match loc.kind {
        MacroCallKind::Attr { invoc_attr_index, .. } => {
            let attr = if loc.def.is_attribute_derive() {
                // for pseudo-derive expansion we actually pass the attribute itself only
                ast::Attr::cast(speculative_args.clone())
            } else {
                // Attributes may have an input token tree, build the subtree and map for this as well
                // then try finding a token id for our token if it is inside this input subtree.
                let item = ast::Item::cast(speculative_args.clone())?;
                collect_attrs(&item)
                    .nth(invoc_attr_index.ast_index())
                    .and_then(|x| Either::left(x.1))
            }?;
            match attr.token_tree() {
                Some(token_tree) => {
                    let mut tree = syntax_node_to_token_tree(token_tree.syntax(), span_map, span);
                    tree.delimiter = tt::Delimiter::invisible_spanned(span);

                    Some(tree)
                }
                _ => None,
            }
        }
        _ => None,
    };

    // Do the actual expansion, we need to directly expand the proc macro due to the attribute args
    // Otherwise the expand query will fetch the non speculative attribute args and pass those instead.
    let mut speculative_expansion = match loc.def.kind {
        MacroDefKind::ProcMacro(expander, _, ast) => {
            let span = db.proc_macro_span(ast);
            tt.delimiter = tt::Delimiter::invisible_spanned(span);
            expander.expand(
                db,
                loc.def.krate,
                loc.krate,
                &tt,
                attr_arg.as_ref(),
                span_with_def_site_ctxt(db, span, actual_macro_call),
                span_with_call_site_ctxt(db, span, actual_macro_call),
                span_with_mixed_site_ctxt(db, span, actual_macro_call),
            )
        }
        MacroDefKind::BuiltInAttr(BuiltinAttrExpander::Derive, _) => {
            pseudo_derive_attr_expansion(&tt, attr_arg.as_ref()?, span)
        }
        MacroDefKind::Declarative(it) => {
            db.decl_macro_expander(loc.krate, it).expand_unhygienic(db, tt, loc.def.krate, span)
        }
        MacroDefKind::BuiltIn(it, _) => {
            it.expand(db, actual_macro_call, &tt, span).map_err(Into::into)
        }
        MacroDefKind::BuiltInDerive(it, ..) => {
            it.expand(db, actual_macro_call, &tt, span).map_err(Into::into)
        }
        MacroDefKind::BuiltInEager(it, _) => {
            it.expand(db, actual_macro_call, &tt, span).map_err(Into::into)
        }
        MacroDefKind::BuiltInAttr(it, _) => it.expand(db, actual_macro_call, &tt, span),
    };

    let expand_to = loc.expand_to();

    fixup::reverse_fixups(&mut speculative_expansion.value, &undo_info);
    let (node, rev_tmap) = token_tree_to_syntax_node(&speculative_expansion.value, expand_to);

    let syntax_node = node.syntax_node();
    let token = rev_tmap
        .ranges_with_span(span_map.span_for_range(token_to_map.text_range()))
        .filter_map(|range| syntax_node.covering_element(range).into_token())
        .min_by_key(|t| {
            // prefer tokens of the same kind and text
            // Note the inversion of the score here, as we want to prefer the first token in case
            // of all tokens having the same score
            (t.kind() != token_to_map.kind()) as u8 + (t.text() != token_to_map.text()) as u8
        })?;
    Some((node.syntax_node(), token))
}

fn ast_id_map(db: &dyn ExpandDatabase, file_id: span::HirFileId) -> triomphe::Arc<AstIdMap> {
    triomphe::Arc::new(AstIdMap::from_source(&db.parse_or_expand(file_id)))
}

fn parse_or_expand(db: &dyn ExpandDatabase, file_id: HirFileId) -> SyntaxNode {
    match file_id.repr() {
        HirFileIdRepr::FileId(file_id) => db.parse(file_id).syntax_node(),
        HirFileIdRepr::MacroFile(macro_file) => {
            db.parse_macro_expansion(macro_file).value.0.syntax_node()
        }
    }
}

fn parse_or_expand_with_err(
    db: &dyn ExpandDatabase,
    file_id: HirFileId,
) -> ExpandResult<Parse<SyntaxNode>> {
    match file_id.repr() {
        HirFileIdRepr::FileId(file_id) => ExpandResult::ok(db.parse(file_id).to_syntax()),
        HirFileIdRepr::MacroFile(macro_file) => {
            db.parse_macro_expansion(macro_file).map(|(it, _)| it)
        }
    }
}

// FIXME: We should verify that the parsed node is one of the many macro node variants we expect
// instead of having it be untyped
fn parse_macro_expansion(
    db: &dyn ExpandDatabase,
    macro_file: MacroFileId,
) -> ExpandResult<(Parse<SyntaxNode>, Arc<ExpansionSpanMap>)> {
    let _p = tracing::span!(tracing::Level::INFO, "parse_macro_expansion").entered();
    let loc = db.lookup_intern_macro_call(macro_file.macro_call_id);
    let expand_to = loc.expand_to();
    let mbe::ValueResult { value: tt, err } = macro_expand(db, macro_file.macro_call_id, loc);

    let (parse, rev_token_map) = token_tree_to_syntax_node(
        match &tt {
            CowArc::Arc(it) => it,
            CowArc::Owned(it) => it,
        },
        expand_to,
    );

    ExpandResult { value: (parse, Arc::new(rev_token_map)), err }
}

fn parse_macro_expansion_error(
    db: &dyn ExpandDatabase,
    macro_call_id: MacroCallId,
) -> ExpandResult<Box<[SyntaxError]>> {
    db.parse_macro_expansion(MacroFileId { macro_call_id })
        .map(|it| it.0.errors().into_boxed_slice())
}

pub(crate) fn parse_with_map(
    db: &dyn ExpandDatabase,
    file_id: HirFileId,
) -> (Parse<SyntaxNode>, SpanMap) {
    match file_id.repr() {
        HirFileIdRepr::FileId(file_id) => {
            (db.parse(file_id).to_syntax(), SpanMap::RealSpanMap(db.real_span_map(file_id)))
        }
        HirFileIdRepr::MacroFile(macro_file) => {
            let (parse, map) = db.parse_macro_expansion(macro_file).value;
            (parse, SpanMap::ExpansionSpanMap(map))
        }
    }
}

// FIXME: for derive attributes, this will return separate copies of the same structures! Though
// they may differ in spans due to differing call sites...
fn macro_arg(
    db: &dyn ExpandDatabase,
    id: MacroCallId,
) -> (Arc<tt::Subtree>, SyntaxFixupUndoInfo, Span) {
    let loc = db.lookup_intern_macro_call(id);

    if let MacroCallLoc {
        def: MacroDefId { kind: MacroDefKind::BuiltInEager(..), .. },
        kind: MacroCallKind::FnLike { eager: Some(eager), .. },
        ..
    } = &loc
    {
        return (eager.arg.clone(), SyntaxFixupUndoInfo::NONE, eager.span);
    }

    let (parse, map) = parse_with_map(db, loc.kind.file_id());
    let root = parse.syntax_node();

    let (censor, item_node, span) = match loc.kind {
        MacroCallKind::FnLike { ast_id, .. } => {
            let node = &ast_id.to_ptr(db).to_node(&root);
            let path_range = node
                .path()
                .map_or_else(|| node.syntax().text_range(), |path| path.syntax().text_range());
            let span = map.span_for_range(path_range);

            let dummy_tt = |kind| {
                (
                    Arc::new(tt::Subtree {
                        delimiter: tt::Delimiter { open: span, close: span, kind },
                        token_trees: Box::default(),
                    }),
                    SyntaxFixupUndoInfo::default(),
                    span,
                )
            };

            let Some(tt) = node.token_tree() else {
                return dummy_tt(tt::DelimiterKind::Invisible);
            };
            let first = tt.left_delimiter_token().map(|it| it.kind()).unwrap_or(T!['(']);
            let last = tt.right_delimiter_token().map(|it| it.kind()).unwrap_or(T![.]);

            let mismatched_delimiters = !matches!(
                (first, last),
                (T!['('], T![')']) | (T!['['], T![']']) | (T!['{'], T!['}'])
            );
            if mismatched_delimiters {
                // Don't expand malformed (unbalanced) macro invocations. This is
                // less than ideal, but trying to expand unbalanced  macro calls
                // sometimes produces pathological, deeply nested code which breaks
                // all kinds of things.
                //
                // So instead, we'll return an empty subtree here
                cov_mark::hit!(issue9358_bad_macro_stack_overflow);

                let kind = match first {
                    _ if loc.def.is_proc_macro() => tt::DelimiterKind::Invisible,
                    T!['('] => tt::DelimiterKind::Parenthesis,
                    T!['['] => tt::DelimiterKind::Bracket,
                    T!['{'] => tt::DelimiterKind::Brace,
                    _ => tt::DelimiterKind::Invisible,
                };
                return dummy_tt(kind);
            }

            let mut tt = mbe::syntax_node_to_token_tree(tt.syntax(), map.as_ref(), span);
            if loc.def.is_proc_macro() {
                // proc macros expect their inputs without parentheses, MBEs expect it with them included
                tt.delimiter.kind = tt::DelimiterKind::Invisible;
            }
            return (Arc::new(tt), SyntaxFixupUndoInfo::NONE, span);
        }
        MacroCallKind::Derive { ast_id, derive_attr_index, .. } => {
            let node = ast_id.to_ptr(db).to_node(&root);
            let censor_derive_input = censor_derive_input(derive_attr_index, &node);
            let item_node = node.into();
            let attr_source = attr_source(derive_attr_index, &item_node);
            // FIXME: This is wrong, this should point to the path of the derive attribute`
            let span =
                map.span_for_range(attr_source.as_ref().and_then(|it| it.path()).map_or_else(
                    || item_node.syntax().text_range(),
                    |it| it.syntax().text_range(),
                ));
            (censor_derive_input, item_node, span)
        }
        MacroCallKind::Attr { ast_id, invoc_attr_index, .. } => {
            let node = ast_id.to_ptr(db).to_node(&root);
            let attr_source = attr_source(invoc_attr_index, &node);
            let span = map.span_for_range(
                attr_source
                    .as_ref()
                    .and_then(|it| it.path())
                    .map_or_else(|| node.syntax().text_range(), |it| it.syntax().text_range()),
            );
            (attr_source.into_iter().map(|it| it.syntax().clone().into()).collect(), node, span)
        }
    };

    let (mut tt, undo_info) = {
        let syntax = item_node.syntax();
        let censor_cfg = cfg_process::process_cfg_attrs(syntax, &loc, db).unwrap_or_default();
        let mut fixups = fixup::fixup_syntax(map.as_ref(), syntax, span);
        fixups.append.retain(|it, _| match it {
            syntax::NodeOrToken::Token(_) => true,
            it => !censor.contains(it) && !censor_cfg.contains(it),
        });
        fixups.remove.extend(censor);
        fixups.remove.extend(censor_cfg);

        (
            mbe::syntax_node_to_token_tree_modified(
                syntax,
                map,
                fixups.append,
                fixups.remove,
                span,
            ),
            fixups.undo_info,
        )
    };

    if loc.def.is_proc_macro() {
        // proc macros expect their inputs without parentheses, MBEs expect it with them included
        tt.delimiter.kind = tt::DelimiterKind::Invisible;
    }

    (Arc::new(tt), undo_info, span)
}

// FIXME: Censoring info should be calculated by the caller! Namely by name resolution
/// Derives expect all `#[derive(..)]` invocations up to (and including) the currently invoked one to be stripped
fn censor_derive_input(derive_attr_index: AttrId, node: &ast::Adt) -> FxHashSet<SyntaxElement> {
    // FIXME: handle `cfg_attr`
    cov_mark::hit!(derive_censoring);
    collect_attrs(node)
        .take(derive_attr_index.ast_index() + 1)
        .filter_map(|(_, attr)| Either::left(attr))
        // FIXME, this resolution should not be done syntactically
        // derive is a proper macro now, no longer builtin
        // But we do not have resolution at this stage, this means
        // we need to know about all macro calls for the given ast item here
        // so we require some kind of mapping...
        .filter(|attr| attr.simple_name().as_deref() == Some("derive"))
        .map(|it| it.syntax().clone().into())
        .collect()
}

/// Attributes expect the invoking attribute to be stripped
fn attr_source(invoc_attr_index: AttrId, node: &ast::Item) -> Option<ast::Attr> {
    // FIXME: handle `cfg_attr`
    cov_mark::hit!(attribute_macro_attr_censoring);
    collect_attrs(node).nth(invoc_attr_index.ast_index()).and_then(|(_, attr)| Either::left(attr))
}

impl TokenExpander {
    fn macro_expander(db: &dyn ExpandDatabase, id: MacroDefId) -> TokenExpander {
        match id.kind {
            MacroDefKind::Declarative(ast_id) => {
                TokenExpander::DeclarativeMacro(db.decl_macro_expander(id.krate, ast_id))
            }
            MacroDefKind::BuiltIn(expander, _) => TokenExpander::BuiltIn(expander),
            MacroDefKind::BuiltInAttr(expander, _) => TokenExpander::BuiltInAttr(expander),
            MacroDefKind::BuiltInDerive(expander, _) => TokenExpander::BuiltInDerive(expander),
            MacroDefKind::BuiltInEager(expander, ..) => TokenExpander::BuiltInEager(expander),
            MacroDefKind::ProcMacro(expander, ..) => TokenExpander::ProcMacro(expander),
        }
    }
}

enum CowArc<T> {
    Arc(Arc<T>),
    Owned(T),
}

fn macro_expand(
    db: &dyn ExpandDatabase,
    macro_call_id: MacroCallId,
    loc: MacroCallLoc,
) -> ExpandResult<CowArc<tt::Subtree>> {
    let _p = tracing::span!(tracing::Level::INFO, "macro_expand").entered();

    let (ExpandResult { value: tt, err }, span) = match loc.def.kind {
        MacroDefKind::ProcMacro(..) => return db.expand_proc_macro(macro_call_id).map(CowArc::Arc),
        _ => {
            let (macro_arg, undo_info, span) = db.macro_arg(macro_call_id);

            let arg = &*macro_arg;
            let res =
                match loc.def.kind {
                    MacroDefKind::Declarative(id) => db
                        .decl_macro_expander(loc.def.krate, id)
                        .expand(db, arg.clone(), macro_call_id, span),
                    MacroDefKind::BuiltIn(it, _) => {
                        it.expand(db, macro_call_id, arg, span).map_err(Into::into)
                    }
                    MacroDefKind::BuiltInDerive(it, _) => {
                        it.expand(db, macro_call_id, arg, span).map_err(Into::into)
                    }
                    MacroDefKind::BuiltInEager(it, _) => {
                        // This might look a bit odd, but we do not expand the inputs to eager macros here.
                        // Eager macros inputs are expanded, well, eagerly when we collect the macro calls.
                        // That kind of expansion uses the ast id map of an eager macros input though which goes through
                        // the HirFileId machinery. As eager macro inputs are assigned a macro file id that query
                        // will end up going through here again, whereas we want to just want to inspect the raw input.
                        // As such we just return the input subtree here.
                        let eager = match &loc.kind {
                            MacroCallKind::FnLike { eager: None, .. } => {
                                return ExpandResult::ok(CowArc::Arc(macro_arg.clone()));
                            }
                            MacroCallKind::FnLike { eager: Some(eager), .. } => Some(&**eager),
                            _ => None,
                        };

                        let mut res = it.expand(db, macro_call_id, arg, span).map_err(Into::into);

                        if let Some(EagerCallInfo { error, .. }) = eager {
                            // FIXME: We should report both errors!
                            res.err = error.clone().or(res.err);
                        }
                        res
                    }
                    MacroDefKind::BuiltInAttr(it, _) => {
                        let mut res = it.expand(db, macro_call_id, arg, span);
                        fixup::reverse_fixups(&mut res.value, &undo_info);
                        res
                    }
                    _ => unreachable!(),
                };
            (ExpandResult { value: res.value, err: res.err }, span)
        }
    };

    // Skip checking token tree limit for include! macro call
    if !loc.def.is_include() {
        // Set a hard limit for the expanded tt
        if let Err(value) = check_tt_count(&tt) {
            return value.map(|()| {
                CowArc::Owned(tt::Subtree {
                    delimiter: tt::Delimiter::invisible_spanned(span),
                    token_trees: Box::new([]),
                })
            });
        }
    }

    ExpandResult { value: CowArc::Owned(tt), err }
}

fn proc_macro_span(db: &dyn ExpandDatabase, ast: AstId<ast::Fn>) -> Span {
    let root = db.parse_or_expand(ast.file_id);
    let ast_id_map = &db.ast_id_map(ast.file_id);
    let span_map = &db.span_map(ast.file_id);

    let node = ast_id_map.get(ast.value).to_node(&root);
    let range = ast::HasName::name(&node)
        .map_or_else(|| node.syntax().text_range(), |name| name.syntax().text_range());
    span_map.span_for_range(range)
}

fn expand_proc_macro(db: &dyn ExpandDatabase, id: MacroCallId) -> ExpandResult<Arc<tt::Subtree>> {
    let loc = db.lookup_intern_macro_call(id);
    let (macro_arg, undo_info, span) = db.macro_arg(id);

    let (expander, ast) = match loc.def.kind {
        MacroDefKind::ProcMacro(expander, _, ast) => (expander, ast),
        _ => unreachable!(),
    };

    let attr_arg = match &loc.kind {
        MacroCallKind::Attr { attr_args: Some(attr_args), .. } => Some(&**attr_args),
        _ => None,
    };

    let ExpandResult { value: mut tt, err } = {
        let span = db.proc_macro_span(ast);
        expander.expand(
            db,
            loc.def.krate,
            loc.krate,
            &macro_arg,
            attr_arg,
            span_with_def_site_ctxt(db, span, id),
            span_with_call_site_ctxt(db, span, id),
            span_with_mixed_site_ctxt(db, span, id),
        )
    };

    // Set a hard limit for the expanded tt
    if let Err(value) = check_tt_count(&tt) {
        return value.map(|()| {
            Arc::new(tt::Subtree {
                delimiter: tt::Delimiter::invisible_spanned(span),
                token_trees: Box::new([]),
            })
        });
    }

    fixup::reverse_fixups(&mut tt, &undo_info);

    ExpandResult { value: Arc::new(tt), err }
}

fn token_tree_to_syntax_node(
    tt: &tt::Subtree,
    expand_to: ExpandTo,
) -> (Parse<SyntaxNode>, ExpansionSpanMap) {
    let entry_point = match expand_to {
        ExpandTo::Statements => mbe::TopEntryPoint::MacroStmts,
        ExpandTo::Items => mbe::TopEntryPoint::MacroItems,
        ExpandTo::Pattern => mbe::TopEntryPoint::Pattern,
        ExpandTo::Type => mbe::TopEntryPoint::Type,
        ExpandTo::Expr => mbe::TopEntryPoint::Expr,
    };
    mbe::token_tree_to_syntax_node(tt, entry_point)
}

fn check_tt_count(tt: &tt::Subtree) -> Result<(), ExpandResult<()>> {
    let count = tt.count();
    if TOKEN_LIMIT.check(count).is_err() {
        Err(ExpandResult {
            value: (),
            err: Some(ExpandError::other(format!(
                "macro invocation exceeds token limit: produced {} tokens, limit is {}",
                count,
                TOKEN_LIMIT.inner(),
            ))),
        })
    } else {
        Ok(())
    }
}

fn setup_syntax_context_root(db: &dyn ExpandDatabase) {
    db.intern_syntax_context(SyntaxContextData::root());
}
