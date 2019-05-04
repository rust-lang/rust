use std::{marker::PhantomData, sync::Arc, hash::{Hash, Hasher}};

use ra_arena::{Arena, RawId, impl_arena_id};
use ra_syntax::{SyntaxNodePtr, TreeArc, SyntaxNode, SourceFile, AstNode, ast};

use crate::{HirFileId, DefDatabase};

/// `AstId` points to an AST node in any file.
///
/// It is stable across reparses, and can be used as salsa key/value.
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
    pub(crate) fn file_id(&self) -> HirFileId {
        self.file_id
    }

    pub(crate) fn to_node(&self, db: &impl DefDatabase) -> TreeArc<N> {
        let syntax_node = db.ast_id_to_node(self.file_id, self.file_ast_id.raw);
        N::cast(&syntax_node).unwrap().to_owned()
    }
}

/// `AstId` points to an AST node in a specific file.
#[derive(Debug)]
pub(crate) struct FileAstId<N: AstNode> {
    raw: ErasedFileAstId,
    _ty: PhantomData<N>,
}

impl<N: AstNode> Clone for FileAstId<N> {
    fn clone(&self) -> FileAstId<N> {
        *self
    }
}
impl<N: AstNode> Copy for FileAstId<N> {}

impl<N: AstNode> PartialEq for FileAstId<N> {
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}
impl<N: AstNode> Eq for FileAstId<N> {}
impl<N: AstNode> Hash for FileAstId<N> {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.raw.hash(hasher);
    }
}

impl<N: AstNode> FileAstId<N> {
    pub(crate) fn with_file_id(self, file_id: HirFileId) -> AstId<N> {
        AstId { file_id, file_ast_id: self }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ErasedFileAstId(RawId);
impl_arena_id!(ErasedFileAstId);

/// Maps items' `SyntaxNode`s to `ErasedFileAstId`s and back.
#[derive(Debug, PartialEq, Eq)]
pub struct AstIdMap {
    arena: Arena<ErasedFileAstId, SyntaxNodePtr>,
}

impl AstIdMap {
    pub(crate) fn ast_id_map_query(db: &impl DefDatabase, file_id: HirFileId) -> Arc<AstIdMap> {
        let source_file = db.hir_parse(file_id);
        Arc::new(AstIdMap::from_source_file(&source_file))
    }

    pub(crate) fn ast_id_to_node_query(
        db: &impl DefDatabase,
        file_id: HirFileId,
        ast_id: ErasedFileAstId,
    ) -> TreeArc<SyntaxNode> {
        let source_file = db.hir_parse(file_id);
        db.ast_id_map(file_id).arena[ast_id].to_node(&source_file).to_owned()
    }

    pub(crate) fn ast_id<N: AstNode>(&self, item: &N) -> FileAstId<N> {
        let ptr = SyntaxNodePtr::new(item.syntax());
        let raw = match self.arena.iter().find(|(_id, i)| **i == ptr) {
            Some((it, _)) => it,
            None => panic!(
                "Can't find {:?} in AstIdMap:\n{:?}",
                item.syntax(),
                self.arena.iter().map(|(_id, i)| i).collect::<Vec<_>>(),
            ),
        };

        FileAstId { raw, _ty: PhantomData }
    }

    fn from_source_file(source_file: &SourceFile) -> AstIdMap {
        let mut res = AstIdMap { arena: Arena::default() };
        // By walking the tree in bread-first order we make sure that parents
        // get lower ids then children. That is, adding a new child does not
        // change parent's id. This means that, say, adding a new function to a
        // trait does not change ids of top-level items, which helps caching.
        bfs(source_file.syntax(), |it| {
            if let Some(module_item) = ast::ModuleItem::cast(it) {
                res.alloc(module_item.syntax());
            } else if let Some(macro_call) = ast::MacroCall::cast(it) {
                res.alloc(macro_call.syntax());
            }
        });
        res
    }

    fn alloc(&mut self, item: &SyntaxNode) -> ErasedFileAstId {
        self.arena.alloc(SyntaxNodePtr::new(item))
    }
}

/// Walks the subtree in bfs order, calling `f` for each node.
fn bfs(node: &SyntaxNode, mut f: impl FnMut(&SyntaxNode)) {
    let mut curr_layer = vec![node];
    let mut next_layer = vec![];
    while !curr_layer.is_empty() {
        curr_layer.drain(..).for_each(|node| {
            next_layer.extend(node.children());
            f(node);
        });
        std::mem::swap(&mut curr_layer, &mut next_layer);
    }
}
