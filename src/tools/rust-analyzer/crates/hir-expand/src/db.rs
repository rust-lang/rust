//! Defines database & queries for macro expansion.

use base_db::{salsa, CrateId, FileId, SourceDatabase};
use either::Either;
use limit::Limit;
use mbe::{syntax_node_to_token_tree, ValueResult};
use rustc_hash::FxHashSet;
use span::SyntaxContextId;
use syntax::{
    ast::{self, HasAttrs},
    AstNode, Parse, SyntaxError, SyntaxNode, SyntaxToken, T,
};
use triomphe::Arc;

use crate::{
    ast_id_map::AstIdMap,
    attrs::collect_attrs,
    builtin_attr_macro::pseudo_derive_attr_expansion,
    builtin_fn_macro::EagerExpander,
    declarative::DeclarativeMacroExpander,
    fixup::{self, reverse_fixups, SyntaxFixupUndoInfo},
    hygiene::{
        span_with_call_site_ctxt, span_with_def_site_ctxt, span_with_mixed_site_ctxt,
        SyntaxContextData,
    },
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

    #[salsa::invoke(AstIdMap::new)]
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
    fn macro_arg(
        &self,
        id: MacroCallId,
    ) -> ValueResult<Option<(Arc<tt::Subtree>, SyntaxFixupUndoInfo)>, Arc<Box<[SyntaxError]>>>;
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

    let span_map = RealSpanMap::absolute(FileId::BOGUS);
    let span_map = SpanMapRef::RealSpanMap(&span_map);

    // Build the subtree and token mapping for the speculative args
    let (mut tt, undo_info) = match loc.kind {
        MacroCallKind::FnLike { .. } => (
            mbe::syntax_node_to_token_tree(speculative_args, span_map, loc.call_site),
            SyntaxFixupUndoInfo::NONE,
        ),
        MacroCallKind::Derive { .. } | MacroCallKind::Attr { .. } => {
            let censor = censor_for_macro_input(&loc, speculative_args);
            let mut fixups = fixup::fixup_syntax(span_map, speculative_args, loc.call_site);
            fixups.append.retain(|it, _| match it {
                syntax::NodeOrToken::Node(it) => !censor.contains(it),
                syntax::NodeOrToken::Token(_) => true,
            });
            fixups.remove.extend(censor);
            (
                mbe::syntax_node_to_token_tree_modified(
                    speculative_args,
                    span_map,
                    fixups.append,
                    fixups.remove,
                    loc.call_site,
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
                    let mut tree =
                        syntax_node_to_token_tree(token_tree.syntax(), span_map, loc.call_site);
                    tree.delimiter = tt::Delimiter::invisible_spanned(loc.call_site);

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
        MacroDefKind::ProcMacro(expander, ..) => {
            tt.delimiter = tt::Delimiter::invisible_spanned(loc.call_site);
            expander.expand(
                db,
                loc.def.krate,
                loc.krate,
                &tt,
                attr_arg.as_ref(),
                span_with_def_site_ctxt(db, loc.def.span, actual_macro_call),
                span_with_call_site_ctxt(db, loc.def.span, actual_macro_call),
                span_with_mixed_site_ctxt(db, loc.def.span, actual_macro_call),
            )
        }
        MacroDefKind::BuiltInAttr(BuiltinAttrExpander::Derive, _) => {
            pseudo_derive_attr_expansion(&tt, attr_arg.as_ref()?, loc.call_site)
        }
        MacroDefKind::BuiltInDerive(expander, ..) => {
            // this cast is a bit sus, can we avoid losing the typedness here?
            let adt = ast::Adt::cast(speculative_args.clone()).unwrap();
            expander.expand(db, actual_macro_call, &adt, span_map)
        }
        MacroDefKind::Declarative(it) => db.decl_macro_expander(loc.krate, it).expand_unhygienic(
            db,
            tt,
            loc.def.krate,
            loc.call_site,
        ),
        MacroDefKind::BuiltIn(it, _) => it.expand(db, actual_macro_call, &tt).map_err(Into::into),
        MacroDefKind::BuiltInEager(it, _) => {
            it.expand(db, actual_macro_call, &tt).map_err(Into::into)
        }
        MacroDefKind::BuiltInAttr(it, _) => it.expand(db, actual_macro_call, &tt),
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
        .map(|it| it.0.errors().to_vec().into_boxed_slice())
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

fn macro_arg(
    db: &dyn ExpandDatabase,
    id: MacroCallId,
    // FIXME: consider the following by putting fixup info into eager call info args
    // ) -> ValueResult<Option<Arc<(tt::Subtree, SyntaxFixupUndoInfo)>>, Arc<Box<[SyntaxError]>>> {
) -> ValueResult<Option<(Arc<tt::Subtree>, SyntaxFixupUndoInfo)>, Arc<Box<[SyntaxError]>>> {
    let mismatched_delimiters = |arg: &SyntaxNode| {
        let first = arg.first_child_or_token().map_or(T![.], |it| it.kind());
        let last = arg.last_child_or_token().map_or(T![.], |it| it.kind());
        let well_formed_tt =
            matches!((first, last), (T!['('], T![')']) | (T!['['], T![']']) | (T!['{'], T!['}']));
        if !well_formed_tt {
            // Don't expand malformed (unbalanced) macro invocations. This is
            // less than ideal, but trying to expand unbalanced  macro calls
            // sometimes produces pathological, deeply nested code which breaks
            // all kinds of things.
            //
            // Some day, we'll have explicit recursion counters for all
            // recursive things, at which point this code might be removed.
            cov_mark::hit!(issue9358_bad_macro_stack_overflow);
            Some(Arc::new(Box::new([SyntaxError::new(
                "unbalanced token tree".to_owned(),
                arg.text_range(),
            )]) as Box<[_]>))
        } else {
            None
        }
    };
    let loc = db.lookup_intern_macro_call(id);
    if let Some(EagerCallInfo { arg, .. }) = matches!(loc.def.kind, MacroDefKind::BuiltInEager(..))
        .then(|| loc.eager.as_deref())
        .flatten()
    {
        ValueResult::ok(Some((arg.clone(), SyntaxFixupUndoInfo::NONE)))
    } else {
        let (parse, map) = parse_with_map(db, loc.kind.file_id());
        let root = parse.syntax_node();

        let syntax = match loc.kind {
            MacroCallKind::FnLike { ast_id, .. } => {
                let node = &ast_id.to_ptr(db).to_node(&root);
                let offset = node.syntax().text_range().start();
                match node.token_tree() {
                    Some(tt) => {
                        let tt = tt.syntax();
                        if let Some(e) = mismatched_delimiters(tt) {
                            return ValueResult::only_err(e);
                        }
                        tt.clone()
                    }
                    None => {
                        return ValueResult::only_err(Arc::new(Box::new([
                            SyntaxError::new_at_offset("missing token tree".to_owned(), offset),
                        ])));
                    }
                }
            }
            MacroCallKind::Derive { ast_id, .. } => {
                ast_id.to_ptr(db).to_node(&root).syntax().clone()
            }
            MacroCallKind::Attr { ast_id, .. } => ast_id.to_ptr(db).to_node(&root).syntax().clone(),
        };
        let (mut tt, undo_info) = match loc.kind {
            MacroCallKind::FnLike { .. } => (
                mbe::syntax_node_to_token_tree(&syntax, map.as_ref(), loc.call_site),
                SyntaxFixupUndoInfo::NONE,
            ),
            MacroCallKind::Derive { .. } | MacroCallKind::Attr { .. } => {
                let censor = censor_for_macro_input(&loc, &syntax);
                let mut fixups = fixup::fixup_syntax(map.as_ref(), &syntax, loc.call_site);
                fixups.append.retain(|it, _| match it {
                    syntax::NodeOrToken::Node(it) => !censor.contains(it),
                    syntax::NodeOrToken::Token(_) => true,
                });
                fixups.remove.extend(censor);
                {
                    let mut tt = mbe::syntax_node_to_token_tree_modified(
                        &syntax,
                        map.as_ref(),
                        fixups.append.clone(),
                        fixups.remove.clone(),
                        loc.call_site,
                    );
                    reverse_fixups(&mut tt, &fixups.undo_info);
                }
                (
                    mbe::syntax_node_to_token_tree_modified(
                        &syntax,
                        map,
                        fixups.append,
                        fixups.remove,
                        loc.call_site,
                    ),
                    fixups.undo_info,
                )
            }
        };

        if loc.def.is_proc_macro() {
            // proc macros expect their inputs without parentheses, MBEs expect it with them included
            tt.delimiter.kind = tt::DelimiterKind::Invisible;
        }

        if matches!(loc.def.kind, MacroDefKind::BuiltInEager(..)) {
            match parse.errors() {
                [] => ValueResult::ok(Some((Arc::new(tt), undo_info))),
                errors => ValueResult::new(
                    Some((Arc::new(tt), undo_info)),
                    // Box::<[_]>::from(res.errors()), not stable yet
                    Arc::new(errors.to_vec().into_boxed_slice()),
                ),
            }
        } else {
            ValueResult::ok(Some((Arc::new(tt), undo_info)))
        }
    }
}

// FIXME: Censoring info should be calculated by the caller! Namely by name resolution
/// Certain macro calls expect some nodes in the input to be preprocessed away, namely:
/// - derives expect all `#[derive(..)]` invocations up to the currently invoked one to be stripped
/// - attributes expect the invoking attribute to be stripped
fn censor_for_macro_input(loc: &MacroCallLoc, node: &SyntaxNode) -> FxHashSet<SyntaxNode> {
    // FIXME: handle `cfg_attr`
    (|| {
        let censor = match loc.kind {
            MacroCallKind::FnLike { .. } => return None,
            MacroCallKind::Derive { derive_attr_index, .. } => {
                cov_mark::hit!(derive_censoring);
                ast::Item::cast(node.clone())?
                    .attrs()
                    .take(derive_attr_index.ast_index() + 1)
                    // FIXME, this resolution should not be done syntactically
                    // derive is a proper macro now, no longer builtin
                    // But we do not have resolution at this stage, this means
                    // we need to know about all macro calls for the given ast item here
                    // so we require some kind of mapping...
                    .filter(|attr| attr.simple_name().as_deref() == Some("derive"))
                    .map(|it| it.syntax().clone())
                    .collect()
            }
            MacroCallKind::Attr { .. } if loc.def.is_attribute_derive() => return None,
            MacroCallKind::Attr { invoc_attr_index, .. } => {
                cov_mark::hit!(attribute_macro_attr_censoring);
                collect_attrs(&ast::Item::cast(node.clone())?)
                    .nth(invoc_attr_index.ast_index())
                    .and_then(|x| Either::left(x.1))
                    .map(|attr| attr.syntax().clone())
                    .into_iter()
                    .collect()
            }
        };
        Some(censor)
    })()
    .unwrap_or_default()
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

    let ExpandResult { value: tt, mut err } = match loc.def.kind {
        MacroDefKind::ProcMacro(..) => return db.expand_proc_macro(macro_call_id).map(CowArc::Arc),
        MacroDefKind::BuiltInDerive(expander, ..) => {
            let (root, map) = parse_with_map(db, loc.kind.file_id());
            let root = root.syntax_node();
            let MacroCallKind::Derive { ast_id, .. } = loc.kind else { unreachable!() };
            let node = ast_id.to_ptr(db).to_node(&root);

            // FIXME: Use censoring
            let _censor = censor_for_macro_input(&loc, node.syntax());
            expander.expand(db, macro_call_id, &node, map.as_ref())
        }
        _ => {
            let ValueResult { value, err } = db.macro_arg(macro_call_id);
            let Some((macro_arg, undo_info)) = value else {
                return ExpandResult {
                    value: CowArc::Owned(tt::Subtree {
                        delimiter: tt::Delimiter::invisible_spanned(loc.call_site),
                        token_trees: Box::new([]),
                    }),
                    // FIXME: We should make sure to enforce an invariant that invalid macro
                    // calls do not reach this call path!
                    err: Some(ExpandError::other("invalid token tree")),
                };
            };

            let arg = &*macro_arg;
            match loc.def.kind {
                MacroDefKind::Declarative(id) => {
                    db.decl_macro_expander(loc.def.krate, id).expand(db, arg.clone(), macro_call_id)
                }
                MacroDefKind::BuiltIn(it, _) => {
                    it.expand(db, macro_call_id, arg).map_err(Into::into)
                }
                // This might look a bit odd, but we do not expand the inputs to eager macros here.
                // Eager macros inputs are expanded, well, eagerly when we collect the macro calls.
                // That kind of expansion uses the ast id map of an eager macros input though which goes through
                // the HirFileId machinery. As eager macro inputs are assigned a macro file id that query
                // will end up going through here again, whereas we want to just want to inspect the raw input.
                // As such we just return the input subtree here.
                MacroDefKind::BuiltInEager(..) if loc.eager.is_none() => {
                    return ExpandResult {
                        value: CowArc::Arc(macro_arg.clone()),
                        err: err.map(|err| {
                            let mut buf = String::new();
                            for err in &**err {
                                use std::fmt::Write;
                                _ = write!(buf, "{}, ", err);
                            }
                            buf.pop();
                            buf.pop();
                            ExpandError::other(buf)
                        }),
                    };
                }
                MacroDefKind::BuiltInEager(it, _) => {
                    it.expand(db, macro_call_id, arg).map_err(Into::into)
                }
                MacroDefKind::BuiltInAttr(it, _) => {
                    let mut res = it.expand(db, macro_call_id, arg);
                    fixup::reverse_fixups(&mut res.value, &undo_info);
                    res
                }
                _ => unreachable!(),
            }
        }
    };

    if let Some(EagerCallInfo { error, .. }) = loc.eager.as_deref() {
        // FIXME: We should report both errors!
        err = error.clone().or(err);
    }

    // Skip checking token tree limit for include! macro call
    if !loc.def.is_include() {
        // Set a hard limit for the expanded tt
        if let Err(value) = check_tt_count(&tt) {
            return value.map(|()| {
                CowArc::Owned(tt::Subtree {
                    delimiter: tt::Delimiter::invisible_spanned(loc.call_site),
                    token_trees: Box::new([]),
                })
            });
        }
    }

    ExpandResult { value: CowArc::Owned(tt), err }
}

fn expand_proc_macro(db: &dyn ExpandDatabase, id: MacroCallId) -> ExpandResult<Arc<tt::Subtree>> {
    let loc = db.lookup_intern_macro_call(id);
    let Some((macro_arg, undo_info)) = db.macro_arg(id).value else {
        return ExpandResult {
            value: Arc::new(tt::Subtree {
                delimiter: tt::Delimiter::invisible_spanned(loc.call_site),
                token_trees: Box::new([]),
            }),
            // FIXME: We should make sure to enforce an invariant that invalid macro
            // calls do not reach this call path!
            err: Some(ExpandError::other("invalid token tree")),
        };
    };

    let expander = match loc.def.kind {
        MacroDefKind::ProcMacro(expander, ..) => expander,
        _ => unreachable!(),
    };

    let attr_arg = match &loc.kind {
        MacroCallKind::Attr { attr_args: Some(attr_args), .. } => Some(&**attr_args),
        _ => None,
    };

    let ExpandResult { value: mut tt, err } = expander.expand(
        db,
        loc.def.krate,
        loc.krate,
        &macro_arg,
        attr_arg,
        span_with_def_site_ctxt(db, loc.def.span, id),
        span_with_call_site_ctxt(db, loc.def.span, id),
        span_with_mixed_site_ctxt(db, loc.def.span, id),
    );

    // Set a hard limit for the expanded tt
    if let Err(value) = check_tt_count(&tt) {
        return value.map(|()| {
            Arc::new(tt::Subtree {
                delimiter: tt::Delimiter::invisible_spanned(loc.call_site),
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
