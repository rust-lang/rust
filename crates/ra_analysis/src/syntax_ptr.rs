use ra_syntax::{
    File, TextRange, SyntaxKind, SyntaxNode, SyntaxNodeRef,
    ast::{self, AstNode},
};

use crate::FileId;
use crate::db::SyntaxDatabase;

/// SyntaxPtr is a cheap `Copy` id which identifies a particular syntax node,
/// without retainig syntax tree in memory. You need to explicitelly `resovle`
/// `SyntaxPtr` to get a `SyntaxNode`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SyntaxPtr {
    file_id: FileId,
    local: LocalSyntaxPtr,
}

impl SyntaxPtr {
    pub(crate) fn new(file_id: FileId, node: SyntaxNodeRef) -> SyntaxPtr {
        let local = LocalSyntaxPtr::new(node);
        SyntaxPtr { file_id, local }
    }

    pub(crate) fn resolve(self, db: &impl SyntaxDatabase) -> SyntaxNode {
        let syntax = db.file_syntax(self.file_id);
        self.local.resolve(&syntax)
    }
}


/// A pionter to a syntax node inside a file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LocalSyntaxPtr {
    range: TextRange,
    kind: SyntaxKind,
}

impl LocalSyntaxPtr {
    fn new(node: SyntaxNodeRef) -> LocalSyntaxPtr {
        LocalSyntaxPtr {
            range: node.range(),
            kind: node.kind(),
        }
    }

    fn resolve(self, file: &File) -> SyntaxNode {
        let mut curr = file.syntax();
        loop {
            if curr.range() == self.range && curr.kind() == self.kind {
                return curr.owned();
            }
            curr = curr.children()
                .find(|it| self.range.is_subrange(&it.range()))
                .unwrap_or_else(|| panic!("can't resovle local ptr to SyntaxNode: {:?}", self))
        }
    }
}


#[test]
fn test_local_syntax_ptr() {
    let file = File::parse("struct Foo { f: u32, }");
    let field = file.syntax().descendants().find_map(ast::NamedFieldDef::cast).unwrap();
    let ptr = LocalSyntaxPtr::new(field.syntax());
    let field_syntax = ptr.resolve(&file);
    assert_eq!(field.syntax(), field_syntax);
}
