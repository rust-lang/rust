use ra_syntax::{SourceFileNode, SyntaxKind, SyntaxNode, SyntaxNodeRef, TextRange};

use crate::db::SyntaxDatabase;
use crate::FileId;

pub(crate) fn resolve_syntax_ptr(db: &impl SyntaxDatabase, ptr: SyntaxPtr) -> SyntaxNode {
    let syntax = db.file_syntax(ptr.file_id);
    ptr.local.resolve(&syntax)
}

/// SyntaxPtr is a cheap `Copy` id which identifies a particular syntax node,
/// without retaining syntax tree in memory. You need to explicitly `resolve`
/// `SyntaxPtr` to get a `SyntaxNode`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SyntaxPtr {
    file_id: FileId,
    local: LocalSyntaxPtr,
}

impl SyntaxPtr {
    pub(crate) fn new(file_id: FileId, node: SyntaxNodeRef) -> SyntaxPtr {
        let local = LocalSyntaxPtr::new(node);
        SyntaxPtr { file_id, local }
    }

    pub(crate) fn file_id(self) -> FileId {
        self.file_id
    }
}

/// A pionter to a syntax node inside a file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct LocalSyntaxPtr {
    range: TextRange,
    kind: SyntaxKind,
}

impl LocalSyntaxPtr {
    pub(crate) fn new(node: SyntaxNodeRef) -> LocalSyntaxPtr {
        LocalSyntaxPtr {
            range: node.range(),
            kind: node.kind(),
        }
    }

    pub(crate) fn resolve(self, file: &SourceFileNode) -> SyntaxNode {
        let mut curr = file.syntax();
        loop {
            if curr.range() == self.range && curr.kind() == self.kind {
                return curr.owned();
            }
            curr = curr
                .children()
                .find(|it| self.range.is_subrange(&it.range()))
                .unwrap_or_else(|| panic!("can't resolve local ptr to SyntaxNode: {:?}", self))
        }
    }

    pub(crate) fn into_global(self, file_id: FileId) -> SyntaxPtr {
        SyntaxPtr {
            file_id,
            local: self,
        }
    }

    // Seems unfortunate to expose
    pub(crate) fn range(self) -> TextRange {
        self.range
    }
}

#[test]
fn test_local_syntax_ptr() {
    use ra_syntax::{ast, AstNode};
    let file = SourceFileNode::parse("struct Foo { f: u32, }");
    let field = file
        .syntax()
        .descendants()
        .find_map(ast::NamedFieldDef::cast)
        .unwrap();
    let ptr = LocalSyntaxPtr::new(field.syntax());
    let field_syntax = ptr.resolve(&file);
    assert_eq!(field.syntax(), field_syntax);
}
