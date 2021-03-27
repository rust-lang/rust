//! Defines database & queries for macro expansion.

use std::sync::Arc;

use base_db::{salsa, SourceDatabase};
use mbe::{ExpandError, ExpandResult, MacroDef, MacroRules};
use parser::FragmentKind;
use syntax::{
    algo::diff,
    ast::{MacroStmts, NameOwner},
    AstNode, GreenNode, Parse,
    SyntaxKind::*,
    SyntaxNode,
};

use crate::{
    ast_id_map::AstIdMap, hygiene::HygieneFrame, BuiltinDeriveExpander, BuiltinFnLikeExpander,
    EagerCallLoc, EagerMacroId, HirFileId, HirFileIdRepr, LazyMacroId, MacroCallId, MacroCallLoc,
    MacroDefId, MacroDefKind, MacroFile, ProcMacroExpander,
};

/// Total limit on the number of tokens produced by any macro invocation.
///
/// If an invocation produces more tokens than this limit, it will not be stored in the database and
/// an error will be emitted.
const TOKEN_LIMIT: usize = 524288;

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum TokenExpander {
    MacroRules(mbe::MacroRules),
    MacroDef(mbe::MacroDef),
    Builtin(BuiltinFnLikeExpander),
    BuiltinDerive(BuiltinDeriveExpander),
    ProcMacro(ProcMacroExpander),
}

impl TokenExpander {
    pub fn expand(
        &self,
        db: &dyn AstDatabase,
        id: LazyMacroId,
        tt: &tt::Subtree,
    ) -> mbe::ExpandResult<tt::Subtree> {
        match self {
            TokenExpander::MacroRules(it) => it.expand(tt),
            TokenExpander::MacroDef(it) => it.expand(tt),
            TokenExpander::Builtin(it) => it.expand(db, id, tt),
            // FIXME switch these to ExpandResult as well
            TokenExpander::BuiltinDerive(it) => it.expand(db, id, tt).into(),
            TokenExpander::ProcMacro(_) => {
                // We store the result in salsa db to prevent non-deterministic behavior in
                // some proc-macro implementation
                // See #4315 for details
                db.expand_proc_macro(id.into()).into()
            }
        }
    }

    pub fn map_id_down(&self, id: tt::TokenId) -> tt::TokenId {
        match self {
            TokenExpander::MacroRules(it) => it.map_id_down(id),
            TokenExpander::MacroDef(it) => it.map_id_down(id),
            TokenExpander::Builtin(..) => id,
            TokenExpander::BuiltinDerive(..) => id,
            TokenExpander::ProcMacro(..) => id,
        }
    }

    pub fn map_id_up(&self, id: tt::TokenId) -> (tt::TokenId, mbe::Origin) {
        match self {
            TokenExpander::MacroRules(it) => it.map_id_up(id),
            TokenExpander::MacroDef(it) => it.map_id_up(id),
            TokenExpander::Builtin(..) => (id, mbe::Origin::Call),
            TokenExpander::BuiltinDerive(..) => (id, mbe::Origin::Call),
            TokenExpander::ProcMacro(..) => (id, mbe::Origin::Call),
        }
    }
}

// FIXME: rename to ExpandDatabase
#[salsa::query_group(AstDatabaseStorage)]
pub trait AstDatabase: SourceDatabase {
    fn ast_id_map(&self, file_id: HirFileId) -> Arc<AstIdMap>;

    #[salsa::transparent]
    fn parse_or_expand(&self, file_id: HirFileId) -> Option<SyntaxNode>;

    #[salsa::interned]
    fn intern_macro(&self, macro_call: MacroCallLoc) -> LazyMacroId;
    fn macro_arg_text(&self, id: MacroCallId) -> Option<GreenNode>;
    #[salsa::transparent]
    fn macro_arg(&self, id: MacroCallId) -> Option<Arc<(tt::Subtree, mbe::TokenMap)>>;
    fn macro_def(&self, id: MacroDefId) -> Option<Arc<(TokenExpander, mbe::TokenMap)>>;
    fn parse_macro_expansion(
        &self,
        macro_file: MacroFile,
    ) -> ExpandResult<Option<(Parse<SyntaxNode>, Arc<mbe::TokenMap>)>>;
    fn macro_expand(&self, macro_call: MacroCallId) -> ExpandResult<Option<Arc<tt::Subtree>>>;

    /// Firewall query that returns the error from the `macro_expand` query.
    fn macro_expand_error(&self, macro_call: MacroCallId) -> Option<ExpandError>;

    #[salsa::interned]
    fn intern_eager_expansion(&self, eager: EagerCallLoc) -> EagerMacroId;

    fn expand_proc_macro(&self, call: MacroCallId) -> Result<tt::Subtree, mbe::ExpandError>;

    fn hygiene_frame(&self, file_id: HirFileId) -> Arc<HygieneFrame>;
}

/// This expands the given macro call, but with different arguments. This is
/// used for completion, where we want to see what 'would happen' if we insert a
/// token. The `token_to_map` mapped down into the expansion, with the mapped
/// token returned.
pub fn expand_hypothetical(
    db: &dyn AstDatabase,
    actual_macro_call: MacroCallId,
    hypothetical_args: &syntax::ast::TokenTree,
    token_to_map: syntax::SyntaxToken,
) -> Option<(SyntaxNode, syntax::SyntaxToken)> {
    let macro_file = MacroFile { macro_call_id: actual_macro_call };
    let (tt, tmap_1) = mbe::syntax_node_to_token_tree(hypothetical_args.syntax()).unwrap();
    let range =
        token_to_map.text_range().checked_sub(hypothetical_args.syntax().text_range().start())?;
    let token_id = tmap_1.token_by_range(range)?;
    let macro_def = expander(db, actual_macro_call)?;
    let (node, tmap_2) =
        parse_macro_with_arg(db, macro_file, Some(std::sync::Arc::new((tt, tmap_1)))).value?;
    let token_id = macro_def.0.map_id_down(token_id);
    let range = tmap_2.range_by_token(token_id)?.by_kind(token_to_map.kind())?;
    let token = node.syntax_node().covering_element(range).into_token()?;
    Some((node.syntax_node(), token))
}

fn ast_id_map(db: &dyn AstDatabase, file_id: HirFileId) -> Arc<AstIdMap> {
    let map =
        db.parse_or_expand(file_id).map_or_else(AstIdMap::default, |it| AstIdMap::from_source(&it));
    Arc::new(map)
}

fn macro_def(db: &dyn AstDatabase, id: MacroDefId) -> Option<Arc<(TokenExpander, mbe::TokenMap)>> {
    match id.kind {
        MacroDefKind::Declarative(ast_id) => match ast_id.to_node(db) {
            syntax::ast::Macro::MacroRules(macro_rules) => {
                let arg = macro_rules.token_tree()?;
                let (tt, tmap) = mbe::ast_to_token_tree(&arg).or_else(|| {
                    log::warn!("fail on macro_rules to token tree: {:#?}", arg);
                    None
                })?;
                let rules = match MacroRules::parse(&tt) {
                    Ok(it) => it,
                    Err(err) => {
                        let name = macro_rules.name().map(|n| n.to_string()).unwrap_or_default();
                        log::warn!("fail on macro_def parse ({}): {:?} {:#?}", name, err, tt);
                        return None;
                    }
                };
                Some(Arc::new((TokenExpander::MacroRules(rules), tmap)))
            }
            syntax::ast::Macro::MacroDef(macro_def) => {
                let arg = macro_def.body()?;
                let (tt, tmap) = mbe::ast_to_token_tree(&arg).or_else(|| {
                    log::warn!("fail on macro_def to token tree: {:#?}", arg);
                    None
                })?;
                let rules = match MacroDef::parse(&tt) {
                    Ok(it) => it,
                    Err(err) => {
                        let name = macro_def.name().map(|n| n.to_string()).unwrap_or_default();
                        log::warn!("fail on macro_def parse ({}): {:?} {:#?}", name, err, tt);
                        return None;
                    }
                };
                Some(Arc::new((TokenExpander::MacroDef(rules), tmap)))
            }
        },
        MacroDefKind::BuiltIn(expander, _) => {
            Some(Arc::new((TokenExpander::Builtin(expander), mbe::TokenMap::default())))
        }
        MacroDefKind::BuiltInDerive(expander, _) => {
            Some(Arc::new((TokenExpander::BuiltinDerive(expander), mbe::TokenMap::default())))
        }
        MacroDefKind::BuiltInEager(..) => None,
        MacroDefKind::ProcMacro(expander, ..) => {
            Some(Arc::new((TokenExpander::ProcMacro(expander), mbe::TokenMap::default())))
        }
    }
}

fn macro_arg_text(db: &dyn AstDatabase, id: MacroCallId) -> Option<GreenNode> {
    let id = match id {
        MacroCallId::LazyMacro(id) => id,
        MacroCallId::EagerMacro(_id) => {
            // FIXME: support macro_arg for eager macro
            return None;
        }
    };
    let loc = db.lookup_intern_macro(id);
    let arg = loc.kind.arg(db)?;
    Some(arg.green())
}

fn macro_arg(db: &dyn AstDatabase, id: MacroCallId) -> Option<Arc<(tt::Subtree, mbe::TokenMap)>> {
    let arg = db.macro_arg_text(id)?;
    let (tt, tmap) = mbe::syntax_node_to_token_tree(&SyntaxNode::new_root(arg))?;
    Some(Arc::new((tt, tmap)))
}

fn macro_expand(db: &dyn AstDatabase, id: MacroCallId) -> ExpandResult<Option<Arc<tt::Subtree>>> {
    macro_expand_with_arg(db, id, None)
}

fn macro_expand_error(db: &dyn AstDatabase, macro_call: MacroCallId) -> Option<ExpandError> {
    db.macro_expand(macro_call).err
}

fn expander(db: &dyn AstDatabase, id: MacroCallId) -> Option<Arc<(TokenExpander, mbe::TokenMap)>> {
    let lazy_id = match id {
        MacroCallId::LazyMacro(id) => id,
        MacroCallId::EagerMacro(_id) => {
            return None;
        }
    };

    let loc = db.lookup_intern_macro(lazy_id);
    let macro_rules = db.macro_def(loc.def)?;
    Some(macro_rules)
}

fn macro_expand_with_arg(
    db: &dyn AstDatabase,
    id: MacroCallId,
    arg: Option<Arc<(tt::Subtree, mbe::TokenMap)>>,
) -> ExpandResult<Option<Arc<tt::Subtree>>> {
    let _p = profile::span("macro_expand");
    let lazy_id = match id {
        MacroCallId::LazyMacro(id) => id,
        MacroCallId::EagerMacro(id) => {
            if arg.is_some() {
                return ExpandResult::str_err(
                    "hypothetical macro expansion not implemented for eager macro".to_owned(),
                );
            } else {
                return ExpandResult {
                    value: Some(db.lookup_intern_eager_expansion(id).subtree),
                    // FIXME: There could be errors here!
                    err: None,
                };
            }
        }
    };

    let loc = db.lookup_intern_macro(lazy_id);
    let macro_arg = match arg.or_else(|| db.macro_arg(id)) {
        Some(it) => it,
        None => return ExpandResult::str_err("Fail to args in to tt::TokenTree".into()),
    };

    let macro_rules = match db.macro_def(loc.def) {
        Some(it) => it,
        None => return ExpandResult::str_err("Fail to find macro definition".into()),
    };
    let ExpandResult { value: tt, err } = macro_rules.0.expand(db, lazy_id, &macro_arg.0);
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
    let lazy_id = match id {
        MacroCallId::LazyMacro(id) => id,
        MacroCallId::EagerMacro(_) => unreachable!(),
    };

    let loc = db.lookup_intern_macro(lazy_id);
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

    expander.expand(db, loc.krate, &macro_arg.0)
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
    parse_macro_with_arg(db, macro_file, None)
}

fn parse_macro_with_arg(
    db: &dyn AstDatabase,
    macro_file: MacroFile,
    arg: Option<Arc<(tt::Subtree, mbe::TokenMap)>>,
) -> ExpandResult<Option<(Parse<SyntaxNode>, Arc<mbe::TokenMap>)>> {
    let macro_call_id = macro_file.macro_call_id;
    let result = if let Some(arg) = arg {
        macro_expand_with_arg(db, macro_call_id, Some(arg))
    } else {
        db.macro_expand(macro_call_id)
    };

    let _p = profile::span("parse_macro_expansion");

    if let Some(err) = &result.err {
        // Note:
        // The final goal we would like to make all parse_macro success,
        // such that the following log will not call anyway.
        match macro_call_id {
            MacroCallId::LazyMacro(id) => {
                let loc: MacroCallLoc = db.lookup_intern_macro(id);
                let node = loc.kind.node(db);

                // collect parent information for warning log
                let parents = std::iter::successors(loc.kind.file_id().call_node(db), |it| {
                    it.file_id.call_node(db)
                })
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
            _ => {
                log::warn!("fail on macro_parse: (reason: {:?})", err);
            }
        }
    }
    let tt = match result.value {
        Some(tt) => tt,
        None => return ExpandResult { value: None, err: result.err },
    };

    let fragment_kind = to_fragment_kind(db, macro_call_id);

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
                return ExpandResult::only_err(err);
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

fn is_self_replicating(from: &SyntaxNode, to: &SyntaxNode) -> bool {
    if diff(from, to).is_empty() {
        return true;
    }
    if let Some(stmts) = MacroStmts::cast(from.clone()) {
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

/// Given a `MacroCallId`, return what `FragmentKind` it belongs to.
/// FIXME: Not completed
fn to_fragment_kind(db: &dyn AstDatabase, id: MacroCallId) -> FragmentKind {
    let lazy_id = match id {
        MacroCallId::LazyMacro(id) => id,
        MacroCallId::EagerMacro(id) => {
            return db.lookup_intern_eager_expansion(id).fragment;
        }
    };
    let syn = db.lookup_intern_macro(lazy_id).kind.node(db).value;

    let parent = match syn.parent() {
        Some(it) => it,
        None => return FragmentKind::Statements,
    };

    match parent.kind() {
        MACRO_ITEMS | SOURCE_FILE => FragmentKind::Items,
        MACRO_STMTS => FragmentKind::Statements,
        ITEM_LIST => FragmentKind::Items,
        LET_STMT => {
            // FIXME: Handle LHS Pattern
            FragmentKind::Expr
        }
        EXPR_STMT => FragmentKind::Statements,
        BLOCK_EXPR => FragmentKind::Statements,
        ARG_LIST => FragmentKind::Expr,
        TRY_EXPR => FragmentKind::Expr,
        TUPLE_EXPR => FragmentKind::Expr,
        PAREN_EXPR => FragmentKind::Expr,
        ARRAY_EXPR => FragmentKind::Expr,
        FOR_EXPR => FragmentKind::Expr,
        PATH_EXPR => FragmentKind::Expr,
        CLOSURE_EXPR => FragmentKind::Expr,
        CONDITION => FragmentKind::Expr,
        BREAK_EXPR => FragmentKind::Expr,
        RETURN_EXPR => FragmentKind::Expr,
        MATCH_EXPR => FragmentKind::Expr,
        MATCH_ARM => FragmentKind::Expr,
        MATCH_GUARD => FragmentKind::Expr,
        RECORD_EXPR_FIELD => FragmentKind::Expr,
        CALL_EXPR => FragmentKind::Expr,
        INDEX_EXPR => FragmentKind::Expr,
        METHOD_CALL_EXPR => FragmentKind::Expr,
        FIELD_EXPR => FragmentKind::Expr,
        AWAIT_EXPR => FragmentKind::Expr,
        CAST_EXPR => FragmentKind::Expr,
        REF_EXPR => FragmentKind::Expr,
        PREFIX_EXPR => FragmentKind::Expr,
        RANGE_EXPR => FragmentKind::Expr,
        BIN_EXPR => FragmentKind::Expr,
        _ => {
            // Unknown , Just guess it is `Items`
            FragmentKind::Items
        }
    }
}
