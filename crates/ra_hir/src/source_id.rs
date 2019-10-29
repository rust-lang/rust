//! FIXME: write short doc here

use std::{
    hash::{Hash, Hasher},
    sync::Arc,
};

pub use hir_def::ast_id_map::{AstIdMap, ErasedFileAstId, FileAstId};
use ra_syntax::{AstNode, SyntaxNode};

use crate::{db::AstDatabase, HirFileId};

/// `AstId` points to an AST node in any file.
///
/// It is stable across reparses, and can be used as salsa key/value.
// FIXME: isn't this just a `Source<FileAstId<N>>` ?
#[derive(Debug)]
pub(crate) struct AstId<N: AstNode> {
    file_id: HirFileId,
    file_ast_id: FileAstId<N>,
}

impl<N: AstNode> Clone for AstId<N> {
    fn clone(&self) -> AstId<N> {
        *self
    }
}
impl<N: AstNode> Copy for AstId<N> {}

impl<N: AstNode> PartialEq for AstId<N> {
    fn eq(&self, other: &Self) -> bool {
        (self.file_id, self.file_ast_id) == (other.file_id, other.file_ast_id)
    }
}
impl<N: AstNode> Eq for AstId<N> {}
impl<N: AstNode> Hash for AstId<N> {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        (self.file_id, self.file_ast_id).hash(hasher);
    }
}

impl<N: AstNode> AstId<N> {
    pub fn new(file_id: HirFileId, file_ast_id: FileAstId<N>) -> AstId<N> {
        AstId { file_id, file_ast_id }
    }

    pub(crate) fn file_id(&self) -> HirFileId {
        self.file_id
    }

    pub(crate) fn to_node(&self, db: &impl AstDatabase) -> N {
        let syntax_node = db.ast_id_to_node(self.file_id, self.file_ast_id.into());
        N::cast(syntax_node).unwrap()
    }
}

pub(crate) fn ast_id_map_query(db: &impl AstDatabase, file_id: HirFileId) -> Arc<AstIdMap> {
    let map = if let Some(node) = db.parse_or_expand(file_id) {
        AstIdMap::from_source(&node)
    } else {
        AstIdMap::default()
    };
    Arc::new(map)
}

pub(crate) fn file_item_query(
    db: &impl AstDatabase,
    file_id: HirFileId,
    ast_id: ErasedFileAstId,
) -> SyntaxNode {
    let node = db.parse_or_expand(file_id).unwrap();
    db.ast_id_map(file_id)[ast_id].to_node(&node)
}
