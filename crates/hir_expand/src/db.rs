//! Defines database & queries for macro expansion.

use std::sync::Arc;

use base_db::{salsa, SourceDatabase};
use mbe::{ExpandError, ExpandResult};
use parser::FragmentKind;
use syntax::{
    algo::diff,
    ast::{self, NameOwner},
    AstNode, GreenNode, Parse, SyntaxNode, SyntaxToken,
};

use crate::{
    ast_id_map::AstIdMap, hygiene::HygieneFrame, input::process_macro_input, BuiltinAttrExpander,
    BuiltinDeriveExpander, BuiltinFnLikeExpander, HirFileId, HirFileIdRepr, MacroCallId,
    MacroCallKind, MacroCallLoc, MacroDefId, MacroDefKind, MacroFile, ProcMacroExpander,
};

/// Total limit on the number of tokens produced by any macro invocation.
///
/// If an invocation produces more tokens than this limit, it will not be stored in the database and
/// an error will be emitted.
const TOKEN_LIMIT: usize = 524288;

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum TokenExpander {
    /// Old-style `macro_rules`.
    MacroRules { mac: mbe::MacroRules, def_site_token_map: mbe::TokenMap },
    /// AKA macros 2.0.
    MacroDef { mac: mbe::MacroDef, def_site_token_map: mbe::TokenMap },
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
        db: &dyn AstDatabase,
        id: MacroCallId,
        tt: &tt::Subtree,
    ) -> mbe::ExpandResult<tt::Subtree> {
        match self {
            TokenExpander::MacroRules { mac, .. } => mac.expand(tt),
            TokenExpander::MacroDef { mac, .. } => mac.expand(tt),
            TokenExpander::Builtin(it) => it.expand(db, id, tt),
            // FIXME switch these to ExpandResult as well
            TokenExpander::BuiltinAttr(it) => it.expand(db, id, tt).into(),
            TokenExpander::BuiltinDerive(it) => it.expand(db, id, tt).into(),
            TokenExpander::ProcMacro(_) => {
                // We store the result in salsa db to prevent non-deterministic behavior in
                // some proc-macro implementation
                // See #4315 for details
                db.expand_proc_macro(id).into()
            }
        }
    }

    pub(crate) fn map_id_down(&self, id: tt::TokenId) -> tt::TokenId {
        match self {
            TokenExpander::MacroRules { mac, .. } => mac.map_id_down(id),
            TokenExpander::MacroDef { mac, .. } => mac.map_id_down(id),
            TokenExpander::Builtin(..)
            | TokenExpander::BuiltinAttr(..)
            | TokenExpander::BuiltinDerive(..)
            | TokenExpander::ProcMacro(..) => id,
        }
    }

    pub(crate) fn map_id_up(&self, id: tt::TokenId) -> (tt::TokenId, mbe::Origin) {
        match self {
            TokenExpander::MacroRules { mac, .. } => mac.map_id_up(id),
            TokenExpander::MacroDef { mac, .. } => mac.map_id_up(id),
            TokenExpander::Builtin(..)
            | TokenExpander::BuiltinAttr(..)
            | TokenExpander::BuiltinDerive(..)
            | TokenExpander::ProcMacro(..) => (id, mbe::Origin::Call),
        }
    }
}

// FIXME: rename to ExpandDatabase
#[salsa::query_group(AstDatabaseStorage)]
pub trait AstDatabase: SourceDatabase {
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
    fn intern_macro(&self, macro_call: MacroCallLoc) -> MacroCallId;

    /// Lowers syntactic macro call to a token tree representation.
    #[salsa::transparent]
    fn macro_arg(&self, id: MacroCallId) -> Option<Arc<(tt::Subtree, mbe::TokenMap)>>;
    /// Extracts syntax node, corresponding to a macro call. That's a firewall
    /// query, only typing in the macro call itself changes the returned
    /// subtree.
    fn macro_arg_text(&self, id: MacroCallId) -> Option<GreenNode>;
    /// Gets the expander for this macro. This compiles declarative macros, and
    /// just fetches procedural ones.
    fn macro_def(&self, id: MacroDefId) -> Option<Arc<TokenExpander>>;

    /// Expand macro call to a token tree. This query is LRUed (we keep 128 or so results in memory)
    fn macro_expand(&self, macro_call: MacroCallId) -> ExpandResult<Option<Arc<tt::Subtree>>>;
    /// Special case of the previous query for procedural macros. We can't LRU
    /// proc macros, since they are not deterministic in general, and
    /// non-determinism breaks salsa in a very, very, very bad way. @edwin0cheng
    /// heroically debugged this once!
    fn expand_proc_macro(&self, call: MacroCallId) -> Result<tt::Subtree, mbe::ExpandError>;
    /// Firewall query that returns the error from the `macro_expand` query.
    fn macro_expand_error(&self, macro_call: MacroCallId) -> Option<ExpandError>;

    fn hygiene_frame(&self, file_id: HirFileId) -> Arc<HygieneFrame>;
}

/// This expands the given macro call, but with different arguments. This is
/// used for completion, where we want to see what 'would happen' if we insert a
/// token. The `token_to_map` mapped down into the expansion, with the mapped
/// token returned.
pub fn expand_speculative(
    db: &dyn AstDatabase,
    actual_macro_call: MacroCallId,
    speculative_args: &ast::TokenTree,
    token_to_map: SyntaxToken,
) -> Option<(SyntaxNode, SyntaxToken)> {
    let (tt, tmap_1) = mbe::syntax_node_to_token_tree(speculative_args.syntax());
    let range =
        token_to_map.text_range().checked_sub(speculative_args.syntax().text_range().start())?;
    let token_id = tmap_1.token_by_range(range)?;

    let macro_def = {
        let loc: MacroCallLoc = db.lookup_intern_macro(actual_macro_call);
        db.macro_def(loc.def)?
    };

    let speculative_expansion = macro_def.expand(db, actual_macro_call, &tt);

    let fragment_kind = macro_fragment_kind(db, actual_macro_call);

    let (node, tmap_2) =
        mbe::token_tree_to_syntax_node(&speculative_expansion.value, fragment_kind).ok()?;

    let token_id = macro_def.map_id_down(token_id);
    let range = tmap_2.range_by_token(token_id, token_to_map.kind())?;
    let token = node.syntax_node().covering_element(range).into_token()?;
    Some((node.syntax_node(), token))
}

fn ast_id_map(db: &dyn AstDatabase, file_id: HirFileId) -> Arc<AstIdMap> {
    let map = db.parse_or_expand(file_id).map(|it| AstIdMap::from_source(&it)).unwrap_or_default();
    Arc::new(map)
}

fn parse_or_expand(db: &dyn AstDatabase, file_id: HirFileId) -> Option<SyntaxNode> {
    match file_id.0 {
        HirFileIdRepr::FileId(file_id) => Some(db.parse(file_id).tree().syntax().clone()),
        HirFileIdRepr::MacroFile(macro_file) => {
            db.parse_macro_expansion(macro_file).value.map(|(it, _)| it.syntax_node())
        }
    }
}

fn parse_macro_expansion(
    db: &dyn AstDatabase,
    macro_file: MacroFile,
) -> ExpandResult<Option<(Parse<SyntaxNode>, Arc<mbe::TokenMap>)>> {
    let _p = profile::span("parse_macro_expansion");
    let result = db.macro_expand(macro_file.macro_call_id);

    if let Some(err) = &result.err {
        // Note:
        // The final goal we would like to make all parse_macro success,
        // such that the following log will not call anyway.
        let loc: MacroCallLoc = db.lookup_intern_macro(macro_file.macro_call_id);
        let node = loc.kind.to_node(db);

        // collect parent information for warning log
        let parents =
            std::iter::successors(loc.kind.file_id().call_node(db), |it| it.file_id.call_node(db))
                .map(|n| format!("{:#}", n.value))
                .collect::<Vec<_>>()
                .join("\n");

        log::warn!(
            "fail on macro_parse: (reason: {:?} macro_call: {:#}) parents: {}",
            err,
            node.value,
            parents
        );
    }
    let tt = match result.value {
        Some(tt) => tt,
        None => return ExpandResult { value: None, err: result.err },
    };

    let fragment_kind = macro_fragment_kind(db, macro_file.macro_call_id);

    log::debug!("expanded = {}", tt.as_debug_string());
    log::debug!("kind = {:?}", fragment_kind);

    let (parse, rev_token_map) = match mbe::token_tree_to_syntax_node(&tt, fragment_kind) {
        Ok(it) => it,
        Err(err) => {
            log::debug!(
                "failed to parse expanstion to {:?} = {}",
                fragment_kind,
                tt.as_debug_string()
            );
            return ExpandResult::only_err(err);
        }
    };

    match result.err {
        Some(err) => {
            // Safety check for recursive identity macro.
            let node = parse.syntax_node();
            let file: HirFileId = macro_file.into();
            let call_node = match file.call_node(db) {
                Some(it) => it,
                None => {
                    return ExpandResult::only_err(err);
                }
            };
            if is_self_replicating(&node, &call_node.value) {
                ExpandResult::only_err(err)
            } else {
                ExpandResult { value: Some((parse, Arc::new(rev_token_map))), err: Some(err) }
            }
        }
        None => {
            log::debug!("parse = {:?}", parse.syntax_node().kind());
            ExpandResult { value: Some((parse, Arc::new(rev_token_map))), err: None }
        }
    }
}

fn macro_arg(db: &dyn AstDatabase, id: MacroCallId) -> Option<Arc<(tt::Subtree, mbe::TokenMap)>> {
    let arg = db.macro_arg_text(id)?;
    let (mut tt, tmap) = mbe::syntax_node_to_token_tree(&SyntaxNode::new_root(arg));

    let loc: MacroCallLoc = db.lookup_intern_macro(id);
    if loc.def.is_proc_macro() {
        // proc macros expect their inputs without parentheses, MBEs expect it with them included
        tt.delimiter = None;
    }

    Some(Arc::new((tt, tmap)))
}

fn macro_arg_text(db: &dyn AstDatabase, id: MacroCallId) -> Option<GreenNode> {
    let loc = db.lookup_intern_macro(id);
    let arg = loc.kind.arg(db)?;
    let arg = process_macro_input(db, arg, id);
    Some(arg.green().into())
}

fn macro_def(db: &dyn AstDatabase, id: MacroDefId) -> Option<Arc<TokenExpander>> {
    match id.kind {
        MacroDefKind::Declarative(ast_id) => match ast_id.to_node(db) {
            ast::Macro::MacroRules(macro_rules) => {
                let arg = macro_rules.token_tree()?;
                let (tt, def_site_token_map) = mbe::ast_to_token_tree(&arg);
                let mac = match mbe::MacroRules::parse(&tt) {
                    Ok(it) => it,
                    Err(err) => {
                        let name = macro_rules.name().map(|n| n.to_string()).unwrap_or_default();
                        log::warn!("fail on macro_def parse ({}): {:?} {:#?}", name, err, tt);
                        return None;
                    }
                };
                Some(Arc::new(TokenExpander::MacroRules { mac, def_site_token_map }))
            }
            ast::Macro::MacroDef(macro_def) => {
                let arg = macro_def.body()?;
                let (tt, def_site_token_map) = mbe::ast_to_token_tree(&arg);
                let mac = match mbe::MacroDef::parse(&tt) {
                    Ok(it) => it,
                    Err(err) => {
                        let name = macro_def.name().map(|n| n.to_string()).unwrap_or_default();
                        log::warn!("fail on macro_def parse ({}): {:?} {:#?}", name, err, tt);
                        return None;
                    }
                };
                Some(Arc::new(TokenExpander::MacroDef { mac, def_site_token_map }))
            }
        },
        MacroDefKind::BuiltIn(expander, _) => Some(Arc::new(TokenExpander::Builtin(expander))),
        MacroDefKind::BuiltInAttr(expander, _) => {
            Some(Arc::new(TokenExpander::BuiltinAttr(expander)))
        }
        MacroDefKind::BuiltInDerive(expander, _) => {
            Some(Arc::new(TokenExpander::BuiltinDerive(expander)))
        }
        MacroDefKind::BuiltInEager(..) => None,
        MacroDefKind::ProcMacro(expander, ..) => Some(Arc::new(TokenExpander::ProcMacro(expander))),
    }
}

fn macro_expand(db: &dyn AstDatabase, id: MacroCallId) -> ExpandResult<Option<Arc<tt::Subtree>>> {
    macro_expand_with_arg(db, id, None)
}

fn macro_expand_error(db: &dyn AstDatabase, macro_call: MacroCallId) -> Option<ExpandError> {
    db.macro_expand(macro_call).err
}

fn macro_expand_with_arg(
    db: &dyn AstDatabase,
    id: MacroCallId,
    arg: Option<Arc<(tt::Subtree, mbe::TokenMap)>>,
) -> ExpandResult<Option<Arc<tt::Subtree>>> {
    let _p = profile::span("macro_expand");
    let loc: MacroCallLoc = db.lookup_intern_macro(id);
    if let Some(eager) = &loc.eager {
        if arg.is_some() {
            return ExpandResult::str_err(
                "speculative macro expansion not implemented for eager macro".to_owned(),
            );
        } else {
            return ExpandResult {
                value: Some(eager.arg_or_expansion.clone()),
                // FIXME: There could be errors here!
                err: None,
            };
        }
    }

    let macro_arg = match arg.or_else(|| db.macro_arg(id)) {
        Some(it) => it,
        None => return ExpandResult::str_err("Fail to args in to tt::TokenTree".into()),
    };

    let macro_rules = match db.macro_def(loc.def) {
        Some(it) => it,
        None => return ExpandResult::str_err("Fail to find macro definition".into()),
    };
    let ExpandResult { value: tt, err } = macro_rules.expand(db, id, &macro_arg.0);
    // Set a hard limit for the expanded tt
    let count = tt.count();
    if count > TOKEN_LIMIT {
        return ExpandResult::str_err(format!(
            "macro invocation exceeds token limit: produced {} tokens, limit is {}",
            count, TOKEN_LIMIT,
        ));
    }

    ExpandResult { value: Some(Arc::new(tt)), err }
}

fn expand_proc_macro(
    db: &dyn AstDatabase,
    id: MacroCallId,
) -> Result<tt::Subtree, mbe::ExpandError> {
    let loc: MacroCallLoc = db.lookup_intern_macro(id);
    let macro_arg = match db.macro_arg(id) {
        Some(it) => it,
        None => {
            return Err(
                tt::ExpansionError::Unknown("No arguments for proc-macro".to_string()).into()
            )
        }
    };

    let expander = match loc.def.kind {
        MacroDefKind::ProcMacro(expander, ..) => expander,
        _ => unreachable!(),
    };

    let attr_arg = match &loc.kind {
        MacroCallKind::Attr { attr_args, .. } => Some(attr_args),
        _ => None,
    };

    expander.expand(db, loc.krate, &macro_arg.0, attr_arg)
}

fn is_self_replicating(from: &SyntaxNode, to: &SyntaxNode) -> bool {
    if diff(from, to).is_empty() {
        return true;
    }
    if let Some(stmts) = ast::MacroStmts::cast(from.clone()) {
        if stmts.statements().any(|stmt| diff(stmt.syntax(), to).is_empty()) {
            return true;
        }
        if let Some(expr) = stmts.expr() {
            if diff(expr.syntax(), to).is_empty() {
                return true;
            }
        }
    }
    false
}

fn hygiene_frame(db: &dyn AstDatabase, file_id: HirFileId) -> Arc<HygieneFrame> {
    Arc::new(HygieneFrame::new(db, file_id))
}

fn macro_fragment_kind(db: &dyn AstDatabase, id: MacroCallId) -> FragmentKind {
    let loc: MacroCallLoc = db.lookup_intern_macro(id);
    loc.kind.fragment_kind()
}
