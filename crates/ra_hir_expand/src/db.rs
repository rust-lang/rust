//! Defines database & queries for macro expansion.

use std::sync::Arc;

use mbe::MacroRules;
use ra_db::{salsa, SourceDatabase};
use ra_prof::profile;
use ra_syntax::{AstNode, Parse, SyntaxNode};

use crate::{
    ast_id_map::AstIdMap, ExpansionInfo, HirFileId, HirFileIdRepr, MacroCallId, MacroCallLoc,
    MacroDefId, MacroFile, MacroFileKind,
};

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ParseMacroWithInfo {
    pub parsed: Parse<SyntaxNode>,
    pub expansion_info: Arc<ExpansionInfo>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct MacroExpandInfo {
    pub arg_map: Arc<mbe::TokenMap>,
    pub def_map: Arc<mbe::TokenMap>,
}

// FIXME: rename to ExpandDatabase
#[salsa::query_group(AstDatabaseStorage)]
pub trait AstDatabase: SourceDatabase {
    fn ast_id_map(&self, file_id: HirFileId) -> Arc<AstIdMap>;

    #[salsa::transparent]
    fn parse_or_expand(&self, file_id: HirFileId) -> Option<SyntaxNode>;

    #[salsa::interned]
    fn intern_macro(&self, macro_call: MacroCallLoc) -> MacroCallId;
    fn macro_arg(&self, id: MacroCallId) -> Option<(Arc<tt::Subtree>, Arc<mbe::TokenMap>)>;
    fn macro_def(&self, id: MacroDefId) -> Option<(Arc<mbe::MacroRules>, Arc<mbe::TokenMap>)>;
    fn parse_macro(&self, macro_file: MacroFile) -> Option<Parse<SyntaxNode>>;
    fn parse_macro_with_info(&self, macro_file: MacroFile) -> Option<ParseMacroWithInfo>;
    fn macro_expand(
        &self,
        macro_call: MacroCallId,
    ) -> Result<(Arc<tt::Subtree>, MacroExpandInfo), String>;

    fn macro_expansion_info(&self, macro_file: MacroFile) -> Option<Arc<ExpansionInfo>>;
}

pub(crate) fn ast_id_map(db: &dyn AstDatabase, file_id: HirFileId) -> Arc<AstIdMap> {
    let map =
        db.parse_or_expand(file_id).map_or_else(AstIdMap::default, |it| AstIdMap::from_source(&it));
    Arc::new(map)
}

pub(crate) fn macro_def(
    db: &dyn AstDatabase,
    id: MacroDefId,
) -> Option<(Arc<mbe::MacroRules>, Arc<mbe::TokenMap>)> {
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
    Some((Arc::new(rules), Arc::new(tmap)))
}

pub(crate) fn macro_arg(
    db: &dyn AstDatabase,
    id: MacroCallId,
) -> Option<(Arc<tt::Subtree>, Arc<mbe::TokenMap>)> {
    let loc = db.lookup_intern_macro(id);
    let macro_call = loc.ast_id.to_node(db);
    let arg = macro_call.token_tree()?;
    let (tt, tmap) = mbe::ast_to_token_tree(&arg)?;
    Some((Arc::new(tt), Arc::new(tmap)))
}

pub(crate) fn macro_expand(
    db: &dyn AstDatabase,
    id: MacroCallId,
) -> Result<(Arc<tt::Subtree>, MacroExpandInfo), String> {
    let loc = db.lookup_intern_macro(id);
    let macro_arg = db.macro_arg(id).ok_or("Fail to args in to tt::TokenTree")?;

    let macro_rules = db.macro_def(loc.def).ok_or("Fail to find macro definition")?;
    let tt = macro_rules.0.expand(&macro_arg.0).map_err(|err| format!("{:?}", err))?;
    // Set a hard limit for the expanded tt
    let count = tt.count();
    if count > 65536 {
        return Err(format!("Total tokens count exceed limit : count = {}", count));
    }

    Ok((
        Arc::new(tt),
        MacroExpandInfo { arg_map: macro_arg.1.clone(), def_map: macro_rules.1.clone() },
    ))
}

pub(crate) fn parse_or_expand(db: &dyn AstDatabase, file_id: HirFileId) -> Option<SyntaxNode> {
    match file_id.0 {
        HirFileIdRepr::FileId(file_id) => Some(db.parse(file_id).tree().syntax().clone()),
        HirFileIdRepr::MacroFile(macro_file) => {
            db.parse_macro(macro_file).map(|it| it.syntax_node())
        }
    }
}

pub(crate) fn parse_macro(
    db: &dyn AstDatabase,
    macro_file: MacroFile,
) -> Option<Parse<SyntaxNode>> {
    let _p = profile("parse_macro_query");
    db.parse_macro_with_info(macro_file).map(|r| r.parsed)
}

pub(crate) fn parse_macro_with_info(
    db: &dyn AstDatabase,
    macro_file: MacroFile,
) -> Option<ParseMacroWithInfo> {
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
    let res = match macro_file.macro_file_kind {
        MacroFileKind::Items => {
            mbe::token_tree_to_items(&tt.0).ok().map(|(p, map)| (Parse::to_syntax(p), map))
        }
        MacroFileKind::Expr => {
            mbe::token_tree_to_expr(&tt.0).ok().map(|(p, map)| (Parse::to_syntax(p), map))
        }
    };

    res.map(|(parsed, exp_map)| {
        let expand_info = tt.1;
        let loc: MacroCallLoc = db.lookup_intern_macro(macro_call_id);

        let def_start =
            loc.def.ast_id.to_node(db).token_tree().map(|t| t.syntax().text_range().start());
        let arg_start =
            loc.ast_id.to_node(db).token_tree().map(|t| t.syntax().text_range().start());

        let arg_map = arg_start
            .map(|start| exp_map.ranges(&expand_info.arg_map, start))
            .unwrap_or_else(|| Vec::new());
        let def_map = def_start
            .map(|start| exp_map.ranges(&expand_info.def_map, start))
            .unwrap_or_else(|| Vec::new());

        let info = ExpansionInfo { arg_map, def_map };

        ParseMacroWithInfo { parsed, expansion_info: Arc::new(info) }
    })
}

pub(crate) fn macro_expansion_info(
    db: &dyn AstDatabase,
    macro_file: MacroFile,
) -> Option<Arc<ExpansionInfo>> {
    db.parse_macro_with_info(macro_file).map(|res| res.expansion_info.clone())
}
