//! Defines database & queries for macro expansion.

use std::sync::Arc;

use base_db::{salsa, SourceDatabase};
use either::Either;
use limit::Limit;
use mbe::syntax_node_to_token_tree;
use rustc_hash::FxHashSet;
use syntax::{
    ast::{self, HasAttrs, HasDocComments},
    AstNode, GreenNode, Parse, SyntaxNode, SyntaxToken, T,
};

use crate::{
    ast_id_map::AstIdMap, builtin_attr_macro::pseudo_derive_attr_expansion, fixup,
    hygiene::HygieneFrame, tt, BuiltinAttrExpander, BuiltinDeriveExpander, BuiltinFnLikeExpander,
    ExpandError, ExpandResult, ExpandTo, HirFileId, HirFileIdRepr, MacroCallId, MacroCallKind,
    MacroCallLoc, MacroDefId, MacroDefKind, MacroFile, ProcMacroExpander,
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
    DeclarativeMacro { mac: mbe::DeclarativeMacro, def_site_token_map: mbe::TokenMap },
    /// Stuff like `line!` and `file!`.
    Builtin(BuiltinFnLikeExpander),
    /// `global_allocator` and such.
    BuiltinAttr(BuiltinAttrExpander),
    /// `derive(Copy)` and such.
    BuiltinDerive(BuiltinDeriveExpander),
    /// The thing we love the most here in rust-analyzer -- procedural macros.
    ProcMacro(ProcMacroExpander),
}

impl TokenExpander {
    fn expand(
        &self,
        db: &dyn ExpandDatabase,
        id: MacroCallId,
        tt: &tt::Subtree,
    ) -> ExpandResult<tt::Subtree> {
        match self {
            TokenExpander::DeclarativeMacro { mac, .. } => mac.expand(tt).map_err(Into::into),
            TokenExpander::Builtin(it) => it.expand(db, id, tt).map_err(Into::into),
            TokenExpander::BuiltinAttr(it) => it.expand(db, id, tt),
            TokenExpander::BuiltinDerive(it) => it.expand(db, id, tt),
            TokenExpander::ProcMacro(_) => {
                // We store the result in salsa db to prevent non-deterministic behavior in
                // some proc-macro implementation
                // See #4315 for details
                db.expand_proc_macro(id)
            }
        }
    }

    pub(crate) fn map_id_down(&self, id: tt::TokenId) -> tt::TokenId {
        match self {
            TokenExpander::DeclarativeMacro { mac, .. } => mac.map_id_down(id),
            TokenExpander::Builtin(..)
            | TokenExpander::BuiltinAttr(..)
            | TokenExpander::BuiltinDerive(..)
            | TokenExpander::ProcMacro(..) => id,
        }
    }

    pub(crate) fn map_id_up(&self, id: tt::TokenId) -> (tt::TokenId, mbe::Origin) {
        match self {
            TokenExpander::DeclarativeMacro { mac, .. } => mac.map_id_up(id),
            TokenExpander::Builtin(..)
            | TokenExpander::BuiltinAttr(..)
            | TokenExpander::BuiltinDerive(..)
            | TokenExpander::ProcMacro(..) => (id, mbe::Origin::Call),
        }
    }
}

#[salsa::query_group(ExpandDatabaseStorage)]
pub trait ExpandDatabase: SourceDatabase {
    fn ast_id_map(&self, file_id: HirFileId) -> Arc<AstIdMap>;

    /// Main public API -- parses a hir file, not caring whether it's a real
    /// file or a macro expansion.
    #[salsa::transparent]
    fn parse_or_expand(&self, file_id: HirFileId) -> Option<SyntaxNode>;
    /// Implementation for the macro case.
    fn parse_macro_expansion(
        &self,
        macro_file: MacroFile,
    ) -> ExpandResult<Option<(Parse<SyntaxNode>, Arc<mbe::TokenMap>)>>;

    /// Macro ids. That's probably the tricksiest bit in rust-analyzer, and the
    /// reason why we use salsa at all.
    ///
    /// We encode macro definitions into ids of macro calls, this what allows us
    /// to be incremental.
    #[salsa::interned]
    fn intern_macro_call(&self, macro_call: MacroCallLoc) -> MacroCallId;

    /// Lowers syntactic macro call to a token tree representation.
    #[salsa::transparent]
    fn macro_arg(
        &self,
        id: MacroCallId,
    ) -> Option<Arc<(tt::Subtree, mbe::TokenMap, fixup::SyntaxFixupUndoInfo)>>;
    /// Extracts syntax node, corresponding to a macro call. That's a firewall
    /// query, only typing in the macro call itself changes the returned
    /// subtree.
    fn macro_arg_text(&self, id: MacroCallId) -> Option<GreenNode>;
    /// Gets the expander for this macro. This compiles declarative macros, and
    /// just fetches procedural ones.
    fn macro_def(&self, id: MacroDefId) -> Result<Arc<TokenExpander>, mbe::ParseError>;

    /// Expand macro call to a token tree. This query is LRUed (we keep 128 or so results in memory)
    fn macro_expand(&self, macro_call: MacroCallId) -> ExpandResult<Option<Arc<tt::Subtree>>>;
    /// Special case of the previous query for procedural macros. We can't LRU
    /// proc macros, since they are not deterministic in general, and
    /// non-determinism breaks salsa in a very, very, very bad way. @edwin0cheng
    /// heroically debugged this once!
    fn expand_proc_macro(&self, call: MacroCallId) -> ExpandResult<tt::Subtree>;
    /// Firewall query that returns the error from the `macro_expand` query.
    fn macro_expand_error(&self, macro_call: MacroCallId) -> Option<ExpandError>;

    fn hygiene_frame(&self, file_id: HirFileId) -> Arc<HygieneFrame>;
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
    let macro_def = db.macro_def(loc.def).ok()?;
    let token_range = token_to_map.text_range();

    // Build the subtree and token mapping for the speculative args
    let censor = censor_for_macro_input(&loc, speculative_args);
    let mut fixups = fixup::fixup_syntax(speculative_args);
    fixups.replace.extend(censor.into_iter().map(|node| (node.into(), Vec::new())));
    let (mut tt, spec_args_tmap, _) = mbe::syntax_node_to_token_tree_with_modifications(
        speculative_args,
        fixups.token_map,
        fixups.next_id,
        fixups.replace,
        fixups.append,
    );

    let (attr_arg, token_id) = match loc.kind {
        MacroCallKind::Attr { invoc_attr_index, is_derive, .. } => {
            let attr = if is_derive {
                // for pseudo-derive expansion we actually pass the attribute itself only
                ast::Attr::cast(speculative_args.clone())
            } else {
                // Attributes may have an input token tree, build the subtree and map for this as well
                // then try finding a token id for our token if it is inside this input subtree.
                let item = ast::Item::cast(speculative_args.clone())?;
                item.doc_comments_and_attrs()
                    .nth(invoc_attr_index.ast_index())
                    .and_then(Either::left)
            }?;
            match attr.token_tree() {
                Some(token_tree) => {
                    let (mut tree, map) = syntax_node_to_token_tree(attr.token_tree()?.syntax());
                    tree.delimiter = tt::Delimiter::unspecified();

                    let shift = mbe::Shift::new(&tt);
                    shift.shift_all(&mut tree);

                    let token_id = if token_tree.syntax().text_range().contains_range(token_range) {
                        let attr_input_start =
                            token_tree.left_delimiter_token()?.text_range().start();
                        let range = token_range.checked_sub(attr_input_start)?;
                        let token_id = shift.shift(map.token_by_range(range)?);
                        Some(token_id)
                    } else {
                        None
                    };
                    (Some(tree), token_id)
                }
                _ => (None, None),
            }
        }
        _ => (None, None),
    };
    let token_id = match token_id {
        Some(token_id) => token_id,
        // token wasn't inside an attribute input so it has to be in the general macro input
        None => {
            let range = token_range.checked_sub(speculative_args.text_range().start())?;
            let token_id = spec_args_tmap.token_by_range(range)?;
            macro_def.map_id_down(token_id)
        }
    };

    // Do the actual expansion, we need to directly expand the proc macro due to the attribute args
    // Otherwise the expand query will fetch the non speculative attribute args and pass those instead.
    let mut speculative_expansion = match loc.def.kind {
        MacroDefKind::ProcMacro(expander, ..) => {
            tt.delimiter = tt::Delimiter::unspecified();
            expander.expand(db, loc.def.krate, loc.krate, &tt, attr_arg.as_ref())
        }
        MacroDefKind::BuiltInAttr(BuiltinAttrExpander::Derive, _) => {
            pseudo_derive_attr_expansion(&tt, attr_arg.as_ref()?)
        }
        _ => macro_def.expand(db, actual_macro_call, &tt),
    };

    let expand_to = macro_expand_to(db, actual_macro_call);
    fixup::reverse_fixups(&mut speculative_expansion.value, &spec_args_tmap, &fixups.undo_info);
    let (node, rev_tmap) = token_tree_to_syntax_node(&speculative_expansion.value, expand_to);

    let syntax_node = node.syntax_node();
    let token = rev_tmap
        .ranges_by_token(token_id, token_to_map.kind())
        .filter_map(|range| syntax_node.covering_element(range).into_token())
        .min_by_key(|t| {
            // prefer tokens of the same kind and text
            // Note the inversion of the score here, as we want to prefer the first token in case
            // of all tokens having the same score
            (t.kind() != token_to_map.kind()) as u8 + (t.text() != token_to_map.text()) as u8
        })?;
    Some((node.syntax_node(), token))
}

fn ast_id_map(db: &dyn ExpandDatabase, file_id: HirFileId) -> Arc<AstIdMap> {
    let map = db.parse_or_expand(file_id).map(|it| AstIdMap::from_source(&it)).unwrap_or_default();
    Arc::new(map)
}

fn parse_or_expand(db: &dyn ExpandDatabase, file_id: HirFileId) -> Option<SyntaxNode> {
    match file_id.repr() {
        HirFileIdRepr::FileId(file_id) => Some(db.parse(file_id).tree().syntax().clone()),
        HirFileIdRepr::MacroFile(macro_file) => {
            // FIXME: Note how we convert from `Parse` to `SyntaxNode` here,
            // forgetting about parse errors.
            db.parse_macro_expansion(macro_file).value.map(|(it, _)| it.syntax_node())
        }
    }
}

fn parse_macro_expansion(
    db: &dyn ExpandDatabase,
    macro_file: MacroFile,
) -> ExpandResult<Option<(Parse<SyntaxNode>, Arc<mbe::TokenMap>)>> {
    let _p = profile::span("parse_macro_expansion");
    let mbe::ValueResult { value, err } = db.macro_expand(macro_file.macro_call_id);

    if let Some(err) = &err {
        // Note:
        // The final goal we would like to make all parse_macro success,
        // such that the following log will not call anyway.
        let loc: MacroCallLoc = db.lookup_intern_macro_call(macro_file.macro_call_id);
        let node = loc.kind.to_node(db);

        // collect parent information for warning log
        let parents =
            std::iter::successors(loc.kind.file_id().call_node(db), |it| it.file_id.call_node(db))
                .map(|n| format!("{:#}", n.value))
                .collect::<Vec<_>>()
                .join("\n");

        tracing::debug!(
            "fail on macro_parse: (reason: {:?} macro_call: {:#}) parents: {}",
            err,
            node.value,
            parents
        );
    }
    let tt = match value {
        Some(tt) => tt,
        None => return ExpandResult { value: None, err },
    };

    let expand_to = macro_expand_to(db, macro_file.macro_call_id);

    tracing::debug!("expanded = {}", tt.as_debug_string());
    tracing::debug!("kind = {:?}", expand_to);

    let (parse, rev_token_map) = token_tree_to_syntax_node(&tt, expand_to);

    ExpandResult { value: Some((parse, Arc::new(rev_token_map))), err }
}

fn macro_arg(
    db: &dyn ExpandDatabase,
    id: MacroCallId,
) -> Option<Arc<(tt::Subtree, mbe::TokenMap, fixup::SyntaxFixupUndoInfo)>> {
    let arg = db.macro_arg_text(id)?;
    let loc = db.lookup_intern_macro_call(id);

    let node = SyntaxNode::new_root(arg);
    let censor = censor_for_macro_input(&loc, &node);
    let mut fixups = fixup::fixup_syntax(&node);
    fixups.replace.extend(censor.into_iter().map(|node| (node.into(), Vec::new())));
    let (mut tt, tmap, _) = mbe::syntax_node_to_token_tree_with_modifications(
        &node,
        fixups.token_map,
        fixups.next_id,
        fixups.replace,
        fixups.append,
    );

    if loc.def.is_proc_macro() {
        // proc macros expect their inputs without parentheses, MBEs expect it with them included
        tt.delimiter = tt::Delimiter::unspecified();
    }
    Some(Arc::new((tt, tmap, fixups.undo_info)))
}

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
            MacroCallKind::Attr { is_derive: true, .. } => return None,
            MacroCallKind::Attr { invoc_attr_index, .. } => {
                cov_mark::hit!(attribute_macro_attr_censoring);
                ast::Item::cast(node.clone())?
                    .doc_comments_and_attrs()
                    .nth(invoc_attr_index.ast_index())
                    .and_then(Either::left)
                    .map(|attr| attr.syntax().clone())
                    .into_iter()
                    .collect()
            }
        };
        Some(censor)
    })()
    .unwrap_or_default()
}

fn macro_arg_text(db: &dyn ExpandDatabase, id: MacroCallId) -> Option<GreenNode> {
    let loc = db.lookup_intern_macro_call(id);
    let arg = loc.kind.arg(db)?;
    if matches!(loc.kind, MacroCallKind::FnLike { .. }) {
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
            return None;
        }
    }
    Some(arg.green().into())
}

fn macro_def(
    db: &dyn ExpandDatabase,
    id: MacroDefId,
) -> Result<Arc<TokenExpander>, mbe::ParseError> {
    match id.kind {
        MacroDefKind::Declarative(ast_id) => {
            let (mac, def_site_token_map) = match ast_id.to_node(db) {
                ast::Macro::MacroRules(macro_rules) => {
                    let arg = macro_rules
                        .token_tree()
                        .ok_or_else(|| mbe::ParseError::Expected("expected a token tree".into()))?;
                    let (tt, def_site_token_map) = mbe::syntax_node_to_token_tree(arg.syntax());
                    let mac = mbe::DeclarativeMacro::parse_macro_rules(&tt)?;
                    (mac, def_site_token_map)
                }
                ast::Macro::MacroDef(macro_def) => {
                    let arg = macro_def
                        .body()
                        .ok_or_else(|| mbe::ParseError::Expected("expected a token tree".into()))?;
                    let (tt, def_site_token_map) = mbe::syntax_node_to_token_tree(arg.syntax());
                    let mac = mbe::DeclarativeMacro::parse_macro2(&tt)?;
                    (mac, def_site_token_map)
                }
            };
            Ok(Arc::new(TokenExpander::DeclarativeMacro { mac, def_site_token_map }))
        }
        MacroDefKind::BuiltIn(expander, _) => Ok(Arc::new(TokenExpander::Builtin(expander))),
        MacroDefKind::BuiltInAttr(expander, _) => {
            Ok(Arc::new(TokenExpander::BuiltinAttr(expander)))
        }
        MacroDefKind::BuiltInDerive(expander, _) => {
            Ok(Arc::new(TokenExpander::BuiltinDerive(expander)))
        }
        MacroDefKind::BuiltInEager(..) => {
            // FIXME: Return a random error here just to make the types align.
            // This obviously should do something real instead.
            Err(mbe::ParseError::UnexpectedToken("unexpected eager macro".into()))
        }
        MacroDefKind::ProcMacro(expander, ..) => Ok(Arc::new(TokenExpander::ProcMacro(expander))),
    }
}

fn macro_expand(
    db: &dyn ExpandDatabase,
    id: MacroCallId,
) -> ExpandResult<Option<Arc<tt::Subtree>>> {
    let _p = profile::span("macro_expand");
    let loc: MacroCallLoc = db.lookup_intern_macro_call(id);
    if let Some(eager) = &loc.eager {
        return ExpandResult {
            value: Some(eager.arg_or_expansion.clone()),
            // FIXME: There could be errors here!
            err: None,
        };
    }

    let macro_arg = match db.macro_arg(id) {
        Some(it) => it,
        None => {
            return ExpandResult::only_err(ExpandError::Other(
                "Failed to lower macro args to token tree".into(),
            ))
        }
    };

    let expander = match db.macro_def(loc.def) {
        Ok(it) => it,
        // FIXME: This is weird -- we effectively report macro *definition*
        // errors lazily, when we try to expand the macro. Instead, they should
        // be reported at the definition site (when we construct a def map).
        Err(err) => {
            return ExpandResult::only_err(ExpandError::Other(
                format!("invalid macro definition: {err}").into(),
            ))
        }
    };
    let ExpandResult { value: mut tt, err } = expander.expand(db, id, &macro_arg.0);
    // Set a hard limit for the expanded tt
    let count = tt.count();
    if TOKEN_LIMIT.check(count).is_err() {
        return ExpandResult::only_err(ExpandError::Other(
            format!(
                "macro invocation exceeds token limit: produced {} tokens, limit is {}",
                count,
                TOKEN_LIMIT.inner(),
            )
            .into(),
        ));
    }

    fixup::reverse_fixups(&mut tt, &macro_arg.1, &macro_arg.2);

    ExpandResult { value: Some(Arc::new(tt)), err }
}

fn macro_expand_error(db: &dyn ExpandDatabase, macro_call: MacroCallId) -> Option<ExpandError> {
    db.macro_expand(macro_call).err
}

fn expand_proc_macro(db: &dyn ExpandDatabase, id: MacroCallId) -> ExpandResult<tt::Subtree> {
    let loc: MacroCallLoc = db.lookup_intern_macro_call(id);
    let macro_arg = match db.macro_arg(id) {
        Some(it) => it,
        None => {
            return ExpandResult::with_err(
                tt::Subtree::empty(),
                ExpandError::Other("No arguments for proc-macro".into()),
            )
        }
    };

    let expander = match loc.def.kind {
        MacroDefKind::ProcMacro(expander, ..) => expander,
        _ => unreachable!(),
    };

    let attr_arg = match &loc.kind {
        MacroCallKind::Attr { attr_args, .. } => {
            let mut attr_args = attr_args.0.clone();
            mbe::Shift::new(&macro_arg.0).shift_all(&mut attr_args);
            Some(attr_args)
        }
        _ => None,
    };

    expander.expand(db, loc.def.krate, loc.krate, &macro_arg.0, attr_arg.as_ref())
}

fn hygiene_frame(db: &dyn ExpandDatabase, file_id: HirFileId) -> Arc<HygieneFrame> {
    Arc::new(HygieneFrame::new(db, file_id))
}

fn macro_expand_to(db: &dyn ExpandDatabase, id: MacroCallId) -> ExpandTo {
    let loc: MacroCallLoc = db.lookup_intern_macro_call(id);
    loc.kind.expand_to()
}

fn token_tree_to_syntax_node(
    tt: &tt::Subtree,
    expand_to: ExpandTo,
) -> (Parse<SyntaxNode>, mbe::TokenMap) {
    let entry_point = match expand_to {
        ExpandTo::Statements => mbe::TopEntryPoint::MacroStmts,
        ExpandTo::Items => mbe::TopEntryPoint::MacroItems,
        ExpandTo::Pattern => mbe::TopEntryPoint::Pattern,
        ExpandTo::Type => mbe::TopEntryPoint::Type,
        ExpandTo::Expr => mbe::TopEntryPoint::Expr,
    };
    mbe::token_tree_to_syntax_node(tt, entry_point)
}
