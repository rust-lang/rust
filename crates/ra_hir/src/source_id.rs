use std::{marker::PhantomData, sync::Arc};

use ra_arena::{Arena, RawId, impl_arena_id};
use ra_syntax::{SyntaxNodePtr, TreeArc, SyntaxNode, SourceFile, AstNode, ast};

use crate::{HirFileId, DefDatabase};

#[derive(Debug, PartialEq, Eq)]
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

impl<N: AstNode> AstId<N> {
    pub(crate) fn file_id(&self) -> HirFileId {
        self.file_id
    }

    pub(crate) fn to_node(&self, db: &impl DefDatabase) -> TreeArc<N> {
        let syntax_node = db.file_item(self.file_ast_id.raw.with_file_id(self.file_id));
        N::cast(&syntax_node).unwrap().to_owned()
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct FileAstId<N: AstNode> {
    raw: SourceFileItemId,
    _ty: PhantomData<N>,
}

impl<N: AstNode> Clone for FileAstId<N> {
    fn clone(&self) -> FileAstId<N> {
        *self
    }
}

impl<N: AstNode> Copy for FileAstId<N> {}

impl<N: AstNode> FileAstId<N> {
    pub(crate) fn with_file_id(self, file_id: HirFileId) -> AstId<N> {
        AstId { file_id, file_ast_id: self }
    }
}

/// Identifier of item within a specific file. This is stable over reparses, so
/// it's OK to use it as a salsa key/value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct SourceFileItemId(RawId);
impl_arena_id!(SourceFileItemId);

impl SourceFileItemId {
    pub(crate) fn with_file_id(self, file_id: HirFileId) -> SourceItemId {
        SourceItemId { file_id, item_id: self }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SourceItemId {
    pub(crate) file_id: HirFileId,
    pub(crate) item_id: SourceFileItemId,
}

/// Maps items' `SyntaxNode`s to `SourceFileItemId`s and back.
#[derive(Debug, PartialEq, Eq)]
pub struct SourceFileItems {
    file_id: HirFileId,
    arena: Arena<SourceFileItemId, SyntaxNodePtr>,
}

impl SourceFileItems {
    pub(crate) fn file_items_query(
        db: &impl DefDatabase,
        file_id: HirFileId,
    ) -> Arc<SourceFileItems> {
        let source_file = db.hir_parse(file_id);
        Arc::new(SourceFileItems::from_source_file(&source_file, file_id))
    }

    pub(crate) fn file_item_query(
        db: &impl DefDatabase,
        source_item_id: SourceItemId,
    ) -> TreeArc<SyntaxNode> {
        let source_file = db.hir_parse(source_item_id.file_id);
        db.file_items(source_item_id.file_id)[source_item_id.item_id]
            .to_node(&source_file)
            .to_owned()
    }

    pub(crate) fn from_source_file(
        source_file: &SourceFile,
        file_id: HirFileId,
    ) -> SourceFileItems {
        let mut res = SourceFileItems { file_id, arena: Arena::default() };
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

    fn alloc(&mut self, item: &SyntaxNode) -> SourceFileItemId {
        self.arena.alloc(SyntaxNodePtr::new(item))
    }
    pub(crate) fn id_of(&self, file_id: HirFileId, item: &SyntaxNode) -> SourceFileItemId {
        assert_eq!(
            self.file_id, file_id,
            "SourceFileItems: wrong file, expected {:?}, got {:?}",
            self.file_id, file_id
        );
        self.id_of_unchecked(item)
    }
    pub(crate) fn id_of_unchecked(&self, item: &SyntaxNode) -> SourceFileItemId {
        let ptr = SyntaxNodePtr::new(item);
        if let Some((id, _)) = self.arena.iter().find(|(_id, i)| **i == ptr) {
            return id;
        }
        panic!(
            "Can't find {:?} in SourceFileItems:\n{:?}",
            item,
            self.arena.iter().map(|(_id, i)| i).collect::<Vec<_>>(),
        );
    }
    pub(crate) fn ast_id<N: AstNode>(&self, item: &N) -> FileAstId<N> {
        FileAstId { raw: self.id_of_unchecked(item.syntax()), _ty: PhantomData }
    }
}

impl std::ops::Index<SourceFileItemId> for SourceFileItems {
    type Output = SyntaxNodePtr;
    fn index(&self, idx: SourceFileItemId) -> &SyntaxNodePtr {
        &self.arena[idx]
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
