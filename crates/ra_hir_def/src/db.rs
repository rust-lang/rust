use std::sync::Arc;

use ra_db::{salsa, SourceDatabase};
use ra_syntax::{Parse, SyntaxNode};

use crate::{
    ast_id_map::{AstIdMap, ErasedFileAstId},
    expand::{HirFileId, MacroCallId, MacroCallLoc, MacroDefId, MacroFile},
};

#[salsa::query_group(AstDatabaseStorage)]
pub trait AstDatabase: SourceDatabase {
    fn ast_id_map(&self, file_id: HirFileId) -> Arc<AstIdMap>;
    #[salsa::transparent]
    fn ast_id_to_node(&self, file_id: HirFileId, ast_id: ErasedFileAstId) -> SyntaxNode;

    #[salsa::transparent]
    #[salsa::invoke(crate::expand::parse_or_expand_query)]
    fn parse_or_expand(&self, file_id: HirFileId) -> Option<SyntaxNode>;

    #[salsa::interned]
    fn intern_macro(&self, macro_call: MacroCallLoc) -> MacroCallId;
    #[salsa::invoke(crate::expand::macro_arg_query)]
    fn macro_arg(&self, id: MacroCallId) -> Option<Arc<tt::Subtree>>;
    #[salsa::invoke(crate::expand::macro_def_query)]
    fn macro_def(&self, id: MacroDefId) -> Option<Arc<mbe::MacroRules>>;
    #[salsa::invoke(crate::expand::parse_macro_query)]
    fn parse_macro(&self, macro_file: MacroFile) -> Option<Parse<SyntaxNode>>;
    #[salsa::invoke(crate::expand::macro_expand_query)]
    fn macro_expand(&self, macro_call: MacroCallId) -> Result<Arc<tt::Subtree>, String>;
}

pub(crate) fn ast_id_map(db: &impl AstDatabase, file_id: HirFileId) -> Arc<AstIdMap> {
    let map =
        db.parse_or_expand(file_id).map_or_else(AstIdMap::default, |it| AstIdMap::from_source(&it));
    Arc::new(map)
}

pub(crate) fn ast_id_to_node(
    db: &impl AstDatabase,
    file_id: HirFileId,
    ast_id: ErasedFileAstId,
) -> SyntaxNode {
    let node = db.parse_or_expand(file_id).unwrap();
    db.ast_id_map(file_id)[ast_id].to_node(&node)
}
