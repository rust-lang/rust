//! Defines database & queries for macro expansion.

use std::sync::Arc;

use mbe::MacroRules;
use ra_db::{salsa, SourceDatabase};
use ra_parser::FragmentKind;
use ra_prof::profile;
use ra_syntax::{AstNode, Parse, SyntaxNode};

use crate::{
    ast_id_map::AstIdMap, BuiltinFnLikeExpander, HirFileId, HirFileIdRepr, MacroCallId,
    MacroCallLoc, MacroDefId, MacroDefKind, MacroFile, MacroFileKind,
};

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum TokenExpander {
    MacroRules(mbe::MacroRules),
    Builtin(BuiltinFnLikeExpander),
}

impl TokenExpander {
    pub fn expand(
        &self,
        db: &dyn AstDatabase,
        id: MacroCallId,
        tt: &tt::Subtree,
    ) -> Result<tt::Subtree, mbe::ExpandError> {
        match self {
            TokenExpander::MacroRules(it) => it.expand(tt),
            TokenExpander::Builtin(it) => it.expand(db, id, tt),
        }
    }

    pub fn map_id_down(&self, id: tt::TokenId) -> tt::TokenId {
        match self {
            TokenExpander::MacroRules(it) => it.map_id_down(id),
            TokenExpander::Builtin(..) => id,
        }
    }

    pub fn map_id_up(&self, id: tt::TokenId) -> (tt::TokenId, mbe::Origin) {
        match self {
            TokenExpander::MacroRules(it) => it.map_id_up(id),
            TokenExpander::Builtin(..) => (id, mbe::Origin::Def),
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
    fn intern_macro(&self, macro_call: MacroCallLoc) -> MacroCallId;
    fn macro_arg(&self, id: MacroCallId) -> Option<Arc<(tt::Subtree, mbe::TokenMap)>>;
    fn macro_def(&self, id: MacroDefId) -> Option<Arc<(TokenExpander, mbe::TokenMap)>>;
    fn parse_macro(&self, macro_file: MacroFile)
        -> Option<(Parse<SyntaxNode>, Arc<mbe::TokenMap>)>;
    fn macro_expand(&self, macro_call: MacroCallId) -> Result<Arc<tt::Subtree>, String>;
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
            let macro_call = id.ast_id.to_node(db);
            let arg = macro_call.token_tree()?;
            let (tt, tmap) = mbe::ast_to_token_tree(&arg).or_else(|| {
                log::warn!("fail on macro_def to token tree: {:#?}", arg);
                None
            })?;
            let rules = MacroRules::parse(&tt).ok().or_else(|| {
                log::warn!("fail on macro_def parse: {:#?}", tt);
                None
            })?;
            Some(Arc::new((TokenExpander::MacroRules(rules), tmap)))
        }
        MacroDefKind::BuiltIn(expander) => {
            Some(Arc::new((TokenExpander::Builtin(expander.clone()), mbe::TokenMap::default())))
        }
    }
}

pub(crate) fn macro_arg(
    db: &dyn AstDatabase,
    id: MacroCallId,
) -> Option<Arc<(tt::Subtree, mbe::TokenMap)>> {
    let loc = db.lookup_intern_macro(id);
    let macro_call = loc.ast_id.to_node(db);
    let arg = macro_call.token_tree()?;
    let (tt, tmap) = mbe::ast_to_token_tree(&arg)?;
    Some(Arc::new((tt, tmap)))
}

pub(crate) fn macro_expand(
    db: &dyn AstDatabase,
    id: MacroCallId,
) -> Result<Arc<tt::Subtree>, String> {
    let loc = db.lookup_intern_macro(id);
    let macro_arg = db.macro_arg(id).ok_or("Fail to args in to tt::TokenTree")?;

    let macro_rules = db.macro_def(loc.def).ok_or("Fail to find macro definition")?;
    let tt = macro_rules.0.expand(db, id, &macro_arg.0).map_err(|err| format!("{:?}", err))?;
    // Set a hard limit for the expanded tt
    let count = tt.count();
    if count > 65536 {
        return Err(format!("Total tokens count exceed limit : count = {}", count));
    }
    Ok(Arc::new(tt))
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
    let _p = profile("parse_macro_query");

    let macro_call_id = macro_file.macro_call_id;
    let tt = db
        .macro_expand(macro_call_id)
        .map_err(|err| {
            // Note:
            // The final goal we would like to make all parse_macro success,
            // such that the following log will not call anyway.
            log::warn!("fail on macro_parse: (reason: {})", err,);
        })
        .ok()?;

    let fragment_kind = match macro_file.macro_file_kind {
        MacroFileKind::Items => FragmentKind::Items,
        MacroFileKind::Expr => FragmentKind::Expr,
        MacroFileKind::Statements => FragmentKind::Statements,
    };
    let (parse, rev_token_map) = mbe::token_tree_to_syntax_node(&tt, fragment_kind).ok()?;
    Some((parse, Arc::new(rev_token_map)))
}
