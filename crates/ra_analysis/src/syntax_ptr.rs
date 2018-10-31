use std::marker::PhantomData;

use ra_syntax::{
    File, TextRange, SyntaxKind, SyntaxNode, SyntaxNodeRef,
    ast::{self, AstNode},
};

use crate::FileId;
use crate::db::SyntaxDatabase;

salsa::query_group! {
    pub(crate) trait SyntaxPtrDatabase: SyntaxDatabase {
        fn resolve_syntax_ptr(ptr: SyntaxPtr) -> SyntaxNode {
            type ResolveSyntaxPtrQuery;
            storage volatile;
        }
    }
}

fn resolve_syntax_ptr(db: &impl SyntaxDatabase, ptr: SyntaxPtr) -> SyntaxNode {
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
}

struct OwnedAst<T> {
    syntax: SyntaxNode,
    phantom: PhantomData<T>,
}

trait ToAst {
    type Ast;
    fn to_ast(self) -> Self::Ast;
}

impl<'a> ToAst for &'a OwnedAst<ast::FnDef<'static>> {
    type Ast = ast::FnDef<'a>;
    fn to_ast(self) -> ast::FnDef<'a> {
        ast::FnDef::cast(self.syntax.borrowed())
            .unwrap()
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

    pub(crate) fn resolve(self, file: &File) -> SyntaxNode {
        let mut curr = file.syntax();
        loop {
            if curr.range() == self.range && curr.kind() == self.kind {
                return curr.owned();
            }
            curr = curr.children()
                .find(|it| self.range.is_subrange(&it.range()))
                .unwrap_or_else(|| panic!("can't resolve local ptr to SyntaxNode: {:?}", self))
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
