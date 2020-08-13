//! Defines database & queries for macro expansion.

use std::sync::Arc;

use base_db::{salsa, SourceDatabase};
use mbe::{ExpandResult, MacroRules};
use parser::FragmentKind;
use syntax::{algo::diff, AstNode, GreenNode, Parse, SyntaxKind::*, SyntaxNode};

use crate::{
    ast_id_map::AstIdMap, BuiltinDeriveExpander, BuiltinFnLikeExpander, EagerCallLoc, EagerMacroId,
    HirFileId, HirFileIdRepr, LazyMacroId, MacroCallId, MacroCallLoc, MacroDefId, MacroDefKind,
    MacroFile, ProcMacroExpander,
};

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum TokenExpander {
    MacroRules(mbe::MacroRules),
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
            // FIXME switch these to ExpandResult as well
            TokenExpander::Builtin(it) => it.expand(db, id, tt).into(),
            TokenExpander::BuiltinDerive(it) => it.expand(db, id, tt).into(),
            TokenExpander::ProcMacro(_) => {
                // We store the result in salsa db to prevent non-determinisc behavior in
                // some proc-macro implementation
                // See #4315 for details
                db.expand_proc_macro(id.into()).into()
            }
        }
    }

    pub fn map_id_down(&self, id: tt::TokenId) -> tt::TokenId {
        match self {
            TokenExpander::MacroRules(it) => it.map_id_down(id),
            TokenExpander::Builtin(..) => id,
            TokenExpander::BuiltinDerive(..) => id,
            TokenExpander::ProcMacro(..) => id,
        }
    }

    pub fn map_id_up(&self, id: tt::TokenId) -> (tt::TokenId, mbe::Origin) {
        match self {
            TokenExpander::MacroRules(it) => it.map_id_up(id),
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
    fn parse_macro(&self, macro_file: MacroFile)
        -> Option<(Parse<SyntaxNode>, Arc<mbe::TokenMap>)>;
    fn macro_expand(&self, macro_call: MacroCallId) -> (Option<Arc<tt::Subtree>>, Option<String>);

    #[salsa::interned]
    fn intern_eager_expansion(&self, eager: EagerCallLoc) -> EagerMacroId;

    fn expand_proc_macro(&self, call: MacroCallId) -> Result<tt::Subtree, mbe::ExpandError>;
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
        parse_macro_with_arg(db, macro_file, Some(std::sync::Arc::new((tt, tmap_1))))?;
    let token_id = macro_def.0.map_id_down(token_id);
    let range = tmap_2.range_by_token(token_id)?.by_kind(token_to_map.kind())?;
    let token = syntax::algo::find_covering_element(&node.syntax_node(), range).into_token()?;
    Some((node.syntax_node(), token))
}

pub(crate) fn ast_id_map(db: &dyn AstDatabase, file_id: HirFileId) -> Arc<AstIdMap> {
    let map =
        db.parse_or_expand(file_id).map_or_else(AstIdMap::default, |it| AstIdMap::from_source(&it));
    Arc::new(map)
}

pub(crate) fn macro_def(
    db: &dyn AstDatabase,
    id: MacroDefId,
) -> Option<Arc<(TokenExpander, mbe::TokenMap)>> {
    match id.kind {
        MacroDefKind::Declarative => {
            let macro_call = id.ast_id?.to_node(db);
            let arg = macro_call.token_tree()?;
            let (tt, tmap) = mbe::ast_to_token_tree(&arg).or_else(|| {
                log::warn!("fail on macro_def to token tree: {:#?}", arg);
                None
            })?;
            let rules = match MacroRules::parse(&tt) {
                Ok(it) => it,
                Err(err) => {
                    log::warn!("fail on macro_def parse: error: {:#?} {:#?}", err, tt);
                    return None;
                }
            };
            Some(Arc::new((TokenExpander::MacroRules(rules), tmap)))
        }
        MacroDefKind::BuiltIn(expander) => {
            Some(Arc::new((TokenExpander::Builtin(expander), mbe::TokenMap::default())))
        }
        MacroDefKind::BuiltInDerive(expander) => {
            Some(Arc::new((TokenExpander::BuiltinDerive(expander), mbe::TokenMap::default())))
        }
        MacroDefKind::BuiltInEager(_) => None,
        MacroDefKind::CustomDerive(expander) => {
            Some(Arc::new((TokenExpander::ProcMacro(expander), mbe::TokenMap::default())))
        }
    }
}

pub(crate) fn macro_arg_text(db: &dyn AstDatabase, id: MacroCallId) -> Option<GreenNode> {
    let id = match id {
        MacroCallId::LazyMacro(id) => id,
        MacroCallId::EagerMacro(_id) => {
            // FIXME: support macro_arg for eager macro
            return None;
        }
    };
    let loc = db.lookup_intern_macro(id);
    let arg = loc.kind.arg(db)?;
    Some(arg.green().clone())
}

pub(crate) fn macro_arg(
    db: &dyn AstDatabase,
    id: MacroCallId,
) -> Option<Arc<(tt::Subtree, mbe::TokenMap)>> {
    let arg = db.macro_arg_text(id)?;
    let (tt, tmap) = mbe::syntax_node_to_token_tree(&SyntaxNode::new_root(arg))?;
    Some(Arc::new((tt, tmap)))
}

pub(crate) fn macro_expand(
    db: &dyn AstDatabase,
    id: MacroCallId,
) -> (Option<Arc<tt::Subtree>>, Option<String>) {
    macro_expand_with_arg(db, id, None)
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
) -> (Option<Arc<tt::Subtree>>, Option<String>) {
    let lazy_id = match id {
        MacroCallId::LazyMacro(id) => id,
        MacroCallId::EagerMacro(id) => {
            if arg.is_some() {
                return (
                    None,
                    Some("hypothetical macro expansion not implemented for eager macro".to_owned()),
                );
            } else {
                return (Some(db.lookup_intern_eager_expansion(id).subtree), None);
            }
        }
    };

    let loc = db.lookup_intern_macro(lazy_id);
    let macro_arg = match arg.or_else(|| db.macro_arg(id)) {
        Some(it) => it,
        None => return (None, Some("Fail to args in to tt::TokenTree".into())),
    };

    let macro_rules = match db.macro_def(loc.def) {
        Some(it) => it,
        None => return (None, Some("Fail to find macro definition".into())),
    };
    let ExpandResult(tt, err) = macro_rules.0.expand(db, lazy_id, &macro_arg.0);
    // Set a hard limit for the expanded tt
    let count = tt.count();
    if count > 65536 {
        return (None, Some(format!("Total tokens count exceed limit : count = {}", count)));
    }
    (Some(Arc::new(tt)), err.map(|e| format!("{:?}", e)))
}

pub(crate) fn expand_proc_macro(
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
        MacroDefKind::CustomDerive(expander) => expander,
        _ => unreachable!(),
    };

    expander.expand(db, lazy_id, &macro_arg.0)
}

pub(crate) fn parse_or_expand(db: &dyn AstDatabase, file_id: HirFileId) -> Option<SyntaxNode> {
    match file_id.0 {
        HirFileIdRepr::FileId(file_id) => Some(db.parse(file_id).tree().syntax().clone()),
        HirFileIdRepr::MacroFile(macro_file) => {
            db.parse_macro(macro_file).map(|(it, _)| it.syntax_node())
        }
    }
}

pub(crate) fn parse_macro(
    db: &dyn AstDatabase,
    macro_file: MacroFile,
) -> Option<(Parse<SyntaxNode>, Arc<mbe::TokenMap>)> {
    parse_macro_with_arg(db, macro_file, None)
}

pub fn parse_macro_with_arg(
    db: &dyn AstDatabase,
    macro_file: MacroFile,
    arg: Option<Arc<(tt::Subtree, mbe::TokenMap)>>,
) -> Option<(Parse<SyntaxNode>, Arc<mbe::TokenMap>)> {
    let _p = profile::span("parse_macro_query");

    let macro_call_id = macro_file.macro_call_id;
    let (tt, err) = if let Some(arg) = arg {
        macro_expand_with_arg(db, macro_call_id, Some(arg))
    } else {
        db.macro_expand(macro_call_id)
    };
    if let Some(err) = &err {
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
                    "fail on macro_parse: (reason: {} macro_call: {:#}) parents: {}",
                    err,
                    node.value,
                    parents
                );
            }
            _ => {
                log::warn!("fail on macro_parse: (reason: {})", err);
            }
        }
    };
    let tt = tt?;

    let fragment_kind = to_fragment_kind(db, macro_call_id);

    let (parse, rev_token_map) = mbe::token_tree_to_syntax_node(&tt, fragment_kind).ok()?;

    if err.is_none() {
        Some((parse, Arc::new(rev_token_map)))
    } else {
        // FIXME:
        // In future, we should propagate the actual error with recovery information
        // instead of ignore the error here.

        // Safe check for recurisve identity macro
        let node = parse.syntax_node();
        let file: HirFileId = macro_file.into();
        let call_node = file.call_node(db)?;

        if !diff(&node, &call_node.value).is_empty() {
            Some((parse, Arc::new(rev_token_map)))
        } else {
            None
        }
    }
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
        None => {
            // FIXME:
            // If it is root, which means the parent HirFile
            // MacroKindFile must be non-items
            // return expr now.
            return FragmentKind::Expr;
        }
    };

    match parent.kind() {
        MACRO_ITEMS | SOURCE_FILE => FragmentKind::Items,
        ITEM_LIST => FragmentKind::Items,
        LET_STMT => {
            // FIXME: Handle Pattern
            FragmentKind::Expr
        }
        // FIXME: Expand to statements in appropriate positions; HIR lowering needs to handle that
        EXPR_STMT | BLOCK_EXPR => FragmentKind::Expr,
        ARG_LIST => FragmentKind::Expr,
        TRY_EXPR => FragmentKind::Expr,
        TUPLE_EXPR => FragmentKind::Expr,
        PAREN_EXPR => FragmentKind::Expr,

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
