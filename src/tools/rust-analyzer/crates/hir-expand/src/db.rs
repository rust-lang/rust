//! Defines database & queries for macro expansion.

use base_db::{salsa, CrateId, Edition, SourceDatabase};
use either::Either;
use limit::Limit;
use mbe::{syntax_node_to_token_tree, ValueResult};
use rustc_hash::FxHashSet;
use syntax::{
    ast::{self, HasAttrs, HasDocComments},
    AstNode, GreenNode, Parse, SyntaxError, SyntaxNode, SyntaxToken, T,
};
use triomphe::Arc;

use crate::{
    ast_id_map::AstIdMap, builtin_attr_macro::pseudo_derive_attr_expansion,
    builtin_fn_macro::EagerExpander, fixup, hygiene::HygieneFrame, tt, AstId, BuiltinAttrExpander,
    BuiltinDeriveExpander, BuiltinFnLikeExpander, EagerCallInfo, ExpandError, ExpandResult,
    ExpandTo, HirFileId, HirFileIdRepr, MacroCallId, MacroCallKind, MacroCallLoc, MacroDefId,
    MacroDefKind, MacroFile, ProcMacroExpander,
};

/// Total limit on the number of tokens produced by any macro invocation.
///
/// If an invocation produces more tokens than this limit, it will not be stored in the database and
/// an error will be emitted.
///
/// Actual max for `analysis-stats .` at some point: 30672.
static TOKEN_LIMIT: Limit = Limit::new(1_048_576);

#[derive(Debug, Clone, Eq, PartialEq)]
/// Old-style `macro_rules` or the new macros 2.0
pub struct DeclarativeMacroExpander {
    pub mac: mbe::DeclarativeMacro,
    pub def_site_token_map: mbe::TokenMap,
}

impl DeclarativeMacroExpander {
    pub fn expand(&self, tt: tt::Subtree) -> ExpandResult<tt::Subtree> {
        match self.mac.err() {
            Some(e) => ExpandResult::new(
                tt::Subtree::empty(),
                ExpandError::other(format!("invalid macro definition: {e}")),
            ),
            None => self.mac.expand(tt).map_err(Into::into),
        }
    }

    pub fn map_id_down(&self, token_id: tt::TokenId) -> tt::TokenId {
        self.mac.map_id_down(token_id)
    }

    pub fn map_id_up(&self, token_id: tt::TokenId) -> (tt::TokenId, mbe::Origin) {
        self.mac.map_id_up(token_id)
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum TokenExpander {
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
    ProcMacro(ProcMacroExpander),
}

// FIXME: Get rid of these methods
impl TokenExpander {
    pub(crate) fn map_id_down(&self, id: tt::TokenId) -> tt::TokenId {
        match self {
            TokenExpander::DeclarativeMacro(expander) => expander.map_id_down(id),
            TokenExpander::BuiltIn(..)
            | TokenExpander::BuiltInEager(..)
            | TokenExpander::BuiltInAttr(..)
            | TokenExpander::BuiltInDerive(..)
            | TokenExpander::ProcMacro(..) => id,
        }
    }

    pub(crate) fn map_id_up(&self, id: tt::TokenId) -> (tt::TokenId, mbe::Origin) {
        match self {
            TokenExpander::DeclarativeMacro(expander) => expander.map_id_up(id),
            TokenExpander::BuiltIn(..)
            | TokenExpander::BuiltInEager(..)
            | TokenExpander::BuiltInAttr(..)
            | TokenExpander::BuiltInDerive(..)
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
    fn parse_or_expand(&self, file_id: HirFileId) -> SyntaxNode;
    #[salsa::transparent]
    fn parse_or_expand_with_err(&self, file_id: HirFileId) -> ExpandResult<Parse<SyntaxNode>>;
    /// Implementation for the macro case.
    // This query is LRU cached
    fn parse_macro_expansion(
        &self,
        macro_file: MacroFile,
    ) -> ExpandResult<(Parse<SyntaxNode>, Arc<mbe::TokenMap>)>;

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
    ) -> ValueResult<
        Option<Arc<(tt::Subtree, mbe::TokenMap, fixup::SyntaxFixupUndoInfo)>>,
        Arc<Box<[SyntaxError]>>,
    >;
    /// Extracts syntax node, corresponding to a macro call. That's a firewall
    /// query, only typing in the macro call itself changes the returned
    /// subtree.
    fn macro_arg_node(
        &self,
        id: MacroCallId,
    ) -> ValueResult<Option<GreenNode>, Arc<Box<[SyntaxError]>>>;
    /// Fetches the expander for this macro.
    #[salsa::transparent]
    fn macro_expander(&self, id: MacroDefId) -> TokenExpander;
    /// Fetches (and compiles) the expander of this decl macro.
    fn decl_macro_expander(
        &self,
        def_crate: CrateId,
        id: AstId<ast::Macro>,
    ) -> Arc<DeclarativeMacroExpander>;

    /// Expand macro call to a token tree.
    // This query is LRU cached
    fn macro_expand(&self, macro_call: MacroCallId) -> ExpandResult<Arc<tt::Subtree>>;
    #[salsa::invoke(crate::builtin_fn_macro::include_arg_to_tt)]
    fn include_expand(
        &self,
        arg_id: MacroCallId,
    ) -> Result<
        (triomphe::Arc<(::tt::Subtree<::tt::TokenId>, mbe::TokenMap)>, base_db::FileId),
        ExpandError,
    >;
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
        MacroCallKind::Attr { invoc_attr_index, .. } => {
            let attr = if loc.def.is_attribute_derive() {
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
            match loc.def.kind {
                MacroDefKind::Declarative(it) => {
                    db.decl_macro_expander(loc.krate, it).map_id_down(token_id)
                }
                _ => token_id,
            }
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
        MacroDefKind::BuiltInDerive(expander, ..) => {
            // this cast is a bit sus, can we avoid losing the typedness here?
            let adt = ast::Adt::cast(speculative_args.clone()).unwrap();
            expander.expand(db, actual_macro_call, &adt, &spec_args_tmap)
        }
        MacroDefKind::Declarative(it) => db.decl_macro_expander(loc.krate, it).expand(tt),
        MacroDefKind::BuiltIn(it, _) => it.expand(db, actual_macro_call, &tt).map_err(Into::into),
        MacroDefKind::BuiltInEager(it, _) => {
            it.expand(db, actual_macro_call, &tt).map_err(Into::into)
        }
        MacroDefKind::BuiltInAttr(it, _) => it.expand(db, actual_macro_call, &tt),
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
    Arc::new(AstIdMap::from_source(&db.parse_or_expand(file_id)))
}

fn parse_or_expand(db: &dyn ExpandDatabase, file_id: HirFileId) -> SyntaxNode {
    match file_id.repr() {
        HirFileIdRepr::FileId(file_id) => db.parse(file_id).tree().syntax().clone(),
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

fn parse_macro_expansion(
    db: &dyn ExpandDatabase,
    macro_file: MacroFile,
) -> ExpandResult<(Parse<SyntaxNode>, Arc<mbe::TokenMap>)> {
    let _p = profile::span("parse_macro_expansion");
    let mbe::ValueResult { value: tt, err } = db.macro_expand(macro_file.macro_call_id);

    let expand_to = macro_expand_to(db, macro_file.macro_call_id);

    tracing::debug!("expanded = {}", tt.as_debug_string());
    tracing::debug!("kind = {:?}", expand_to);

    let (parse, rev_token_map) = token_tree_to_syntax_node(&tt, expand_to);

    ExpandResult { value: (parse, Arc::new(rev_token_map)), err }
}

fn parse_macro_expansion_error(
    db: &dyn ExpandDatabase,
    macro_call_id: MacroCallId,
) -> ExpandResult<Box<[SyntaxError]>> {
    db.parse_macro_expansion(MacroFile { macro_call_id })
        .map(|it| it.0.errors().to_vec().into_boxed_slice())
}

fn macro_arg(
    db: &dyn ExpandDatabase,
    id: MacroCallId,
) -> ValueResult<
    Option<Arc<(tt::Subtree, mbe::TokenMap, fixup::SyntaxFixupUndoInfo)>>,
    Arc<Box<[SyntaxError]>>,
> {
    let loc = db.lookup_intern_macro_call(id);

    if let Some(EagerCallInfo { arg, arg_id: _, error: _ }) = loc.eager.as_deref() {
        return ValueResult::ok(Some(Arc::new((arg.0.clone(), arg.1.clone(), Default::default()))));
    }

    let ValueResult { value, err } = db.macro_arg_node(id);
    let Some(arg) = value else {
        return ValueResult { value: None, err };
    };

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
    let val = Some(Arc::new((tt, tmap, fixups.undo_info)));
    match err {
        Some(err) => ValueResult::new(val, err),
        None => ValueResult::ok(val),
    }
}

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

fn macro_arg_node(
    db: &dyn ExpandDatabase,
    id: MacroCallId,
) -> ValueResult<Option<GreenNode>, Arc<Box<[SyntaxError]>>> {
    let err = || -> Arc<Box<[_]>> {
        Arc::new(Box::new([SyntaxError::new_at_offset(
            "invalid macro call".to_owned(),
            syntax::TextSize::from(0),
        )]))
    };
    let loc = db.lookup_intern_macro_call(id);
    let arg = if let MacroDefKind::BuiltInEager(..) = loc.def.kind {
        let res = if let Some(EagerCallInfo { arg, .. }) = loc.eager.as_deref() {
            Some(mbe::token_tree_to_syntax_node(&arg.0, mbe::TopEntryPoint::Expr).0)
        } else {
            loc.kind
                .arg(db)
                .and_then(|arg| ast::TokenTree::cast(arg.value))
                .map(|tt| tt.reparse_as_expr().to_syntax())
        };

        match res {
            Some(res) if res.errors().is_empty() => res.syntax_node(),
            Some(res) => {
                return ValueResult::new(
                    Some(res.syntax_node().green().into()),
                    // Box::<[_]>::from(res.errors()), not stable yet
                    Arc::new(res.errors().to_vec().into_boxed_slice()),
                );
            }
            None => return ValueResult::only_err(err()),
        }
    } else {
        match loc.kind.arg(db) {
            Some(res) => res.value,
            None => return ValueResult::only_err(err()),
        }
    };
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
            return ValueResult::only_err(Arc::new(Box::new([SyntaxError::new(
                "unbalanced token tree".to_owned(),
                arg.text_range(),
            )])));
        }
    }
    ValueResult::ok(Some(arg.green().into()))
}

fn decl_macro_expander(
    db: &dyn ExpandDatabase,
    def_crate: CrateId,
    id: AstId<ast::Macro>,
) -> Arc<DeclarativeMacroExpander> {
    let is_2021 = db.crate_graph()[def_crate].edition >= Edition::Edition2021;
    let (mac, def_site_token_map) = match id.to_node(db) {
        ast::Macro::MacroRules(macro_rules) => match macro_rules.token_tree() {
            Some(arg) => {
                let (tt, def_site_token_map) = mbe::syntax_node_to_token_tree(arg.syntax());
                let mac = mbe::DeclarativeMacro::parse_macro_rules(&tt, is_2021);
                (mac, def_site_token_map)
            }
            None => (
                mbe::DeclarativeMacro::from_err(
                    mbe::ParseError::Expected("expected a token tree".into()),
                    is_2021,
                ),
                Default::default(),
            ),
        },
        ast::Macro::MacroDef(macro_def) => match macro_def.body() {
            Some(arg) => {
                let (tt, def_site_token_map) = mbe::syntax_node_to_token_tree(arg.syntax());
                let mac = mbe::DeclarativeMacro::parse_macro2(&tt, is_2021);
                (mac, def_site_token_map)
            }
            None => (
                mbe::DeclarativeMacro::from_err(
                    mbe::ParseError::Expected("expected a token tree".into()),
                    is_2021,
                ),
                Default::default(),
            ),
        },
    };
    Arc::new(DeclarativeMacroExpander { mac, def_site_token_map })
}

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

fn macro_expand(db: &dyn ExpandDatabase, id: MacroCallId) -> ExpandResult<Arc<tt::Subtree>> {
    let _p = profile::span("macro_expand");
    let loc = db.lookup_intern_macro_call(id);

    let ExpandResult { value: tt, mut err } = match loc.def.kind {
        MacroDefKind::ProcMacro(..) => return db.expand_proc_macro(id),
        MacroDefKind::BuiltInDerive(expander, ..) => {
            let arg = db.macro_arg_node(id).value.unwrap();

            let node = SyntaxNode::new_root(arg);
            let censor = censor_for_macro_input(&loc, &node);
            let mut fixups = fixup::fixup_syntax(&node);
            fixups.replace.extend(censor.into_iter().map(|node| (node.into(), Vec::new())));
            let (tmap, _) = mbe::syntax_node_to_token_map_with_modifications(
                &node,
                fixups.token_map,
                fixups.next_id,
                fixups.replace,
                fixups.append,
            );

            // this cast is a bit sus, can we avoid losing the typedness here?
            let adt = ast::Adt::cast(node).unwrap();
            let mut res = expander.expand(db, id, &adt, &tmap);
            fixup::reverse_fixups(&mut res.value, &tmap, &fixups.undo_info);
            res
        }
        _ => {
            let ValueResult { value, err } = db.macro_arg(id);
            let Some(macro_arg) = value else {
                return ExpandResult {
                    value: Arc::new(tt::Subtree {
                        delimiter: tt::Delimiter::UNSPECIFIED,
                        token_trees: Vec::new(),
                    }),
                    // FIXME: We should make sure to enforce an invariant that invalid macro
                    // calls do not reach this call path!
                    err: Some(ExpandError::other("invalid token tree")),
                };
            };

            let (arg, arg_tm, undo_info) = &*macro_arg;
            let mut res = match loc.def.kind {
                MacroDefKind::Declarative(id) => {
                    db.decl_macro_expander(loc.def.krate, id).expand(arg.clone())
                }
                MacroDefKind::BuiltIn(it, _) => it.expand(db, id, &arg).map_err(Into::into),
                // This might look a bit odd, but we do not expand the inputs to eager macros here.
                // Eager macros inputs are expanded, well, eagerly when we collect the macro calls.
                // That kind of expansion uses the ast id map of an eager macros input though which goes through
                // the HirFileId machinery. As eager macro inputs are assigned a macro file id that query
                // will end up going through here again, whereas we want to just want to inspect the raw input.
                // As such we just return the input subtree here.
                MacroDefKind::BuiltInEager(..) if loc.eager.is_none() => {
                    let mut arg = arg.clone();
                    fixup::reverse_fixups(&mut arg, arg_tm, undo_info);

                    return ExpandResult {
                        value: Arc::new(arg),
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
                MacroDefKind::BuiltInEager(it, _) => it.expand(db, id, &arg).map_err(Into::into),
                MacroDefKind::BuiltInAttr(it, _) => it.expand(db, id, &arg),
                _ => unreachable!(),
            };
            fixup::reverse_fixups(&mut res.value, arg_tm, undo_info);
            res
        }
    };

    if let Some(EagerCallInfo { error, .. }) = loc.eager.as_deref() {
        // FIXME: We should report both errors!
        err = error.clone().or(err);
    }

    // Set a hard limit for the expanded tt
    if let Err(value) = check_tt_count(&tt) {
        return value;
    }

    ExpandResult { value: Arc::new(tt), err }
}

fn expand_proc_macro(db: &dyn ExpandDatabase, id: MacroCallId) -> ExpandResult<Arc<tt::Subtree>> {
    let loc = db.lookup_intern_macro_call(id);
    let Some(macro_arg) = db.macro_arg(id).value else {
        return ExpandResult {
            value: Arc::new(tt::Subtree {
                delimiter: tt::Delimiter::UNSPECIFIED,
                token_trees: Vec::new(),
            }),
            // FIXME: We should make sure to enforce an invariant that invalid macro
            // calls do not reach this call path!
            err: Some(ExpandError::other("invalid token tree")),
        };
    };

    let (arg_tt, arg_tm, undo_info) = &*macro_arg;

    let expander = match loc.def.kind {
        MacroDefKind::ProcMacro(expander, ..) => expander,
        _ => unreachable!(),
    };

    let attr_arg = match &loc.kind {
        MacroCallKind::Attr { attr_args, .. } => {
            let mut attr_args = attr_args.0.clone();
            mbe::Shift::new(arg_tt).shift_all(&mut attr_args);
            Some(attr_args)
        }
        _ => None,
    };

    let ExpandResult { value: mut tt, err } =
        expander.expand(db, loc.def.krate, loc.krate, arg_tt, attr_arg.as_ref());

    // Set a hard limit for the expanded tt
    if let Err(value) = check_tt_count(&tt) {
        return value;
    }

    fixup::reverse_fixups(&mut tt, arg_tm, undo_info);

    ExpandResult { value: Arc::new(tt), err }
}

fn hygiene_frame(db: &dyn ExpandDatabase, file_id: HirFileId) -> Arc<HygieneFrame> {
    Arc::new(HygieneFrame::new(db, file_id))
}

fn macro_expand_to(db: &dyn ExpandDatabase, id: MacroCallId) -> ExpandTo {
    db.lookup_intern_macro_call(id).expand_to()
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

fn check_tt_count(tt: &tt::Subtree) -> Result<(), ExpandResult<Arc<tt::Subtree>>> {
    let count = tt.count();
    if TOKEN_LIMIT.check(count).is_err() {
        Err(ExpandResult {
            value: Arc::new(tt::Subtree {
                delimiter: tt::Delimiter::UNSPECIFIED,
                token_trees: vec![],
            }),
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
