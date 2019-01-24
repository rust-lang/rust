use std::marker::PhantomData;

use crate::{
    AstNode, SourceFile, SyntaxKind, SyntaxNode, TextRange,
    algo::generate,
};

/// A pointer to a syntax node inside a file. It can be used to remember a
/// specific node across reparses of the same file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SyntaxNodePtr {
    range: TextRange,
    kind: SyntaxKind,
}

impl SyntaxNodePtr {
    pub fn new(node: &SyntaxNode) -> SyntaxNodePtr {
        SyntaxNodePtr {
            range: node.range(),
            kind: node.kind(),
        }
    }

    pub fn to_node(self, source_file: &SourceFile) -> &SyntaxNode {
        generate(Some(source_file.syntax()), |&node| {
            node.children()
                .find(|it| self.range.is_subrange(&it.range()))
        })
        .find(|it| it.range() == self.range && it.kind() == self.kind)
        .unwrap_or_else(|| panic!("can't resolve local ptr to SyntaxNode: {:?}", self))
    }

    pub fn range(self) -> TextRange {
        self.range
    }

    pub fn kind(self) -> SyntaxKind {
        self.kind
    }
}

/// Like `SyntaxNodePtr`, but remembers the type of node
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct AstPtr<N: AstNode> {
    raw: SyntaxNodePtr,
    _ty: PhantomData<N>,
}

impl<N: AstNode> Copy for AstPtr<N> {}
impl<N: AstNode> Clone for AstPtr<N> {
    fn clone(&self) -> AstPtr<N> {
        *self
    }
}

impl<N: AstNode> AstPtr<N> {
    pub fn new(node: &N) -> AstPtr<N> {
        AstPtr {
            raw: SyntaxNodePtr::new(node.syntax()),
            _ty: PhantomData,
        }
    }

    pub fn to_node(self, source_file: &SourceFile) -> &N {
        let syntax_node = self.raw.to_node(source_file);
        N::cast(syntax_node).unwrap()
    }

    pub fn syntax_node_ptr(self) -> SyntaxNodePtr {
        self.raw
    }
}

#[test]
fn test_local_syntax_ptr() {
    use crate::{ast, AstNode};

    let file = SourceFile::parse("struct Foo { f: u32, }");
    let field = file
        .syntax()
        .descendants()
        .find_map(ast::NamedFieldDef::cast)
        .unwrap();
    let ptr = SyntaxNodePtr::new(field.syntax());
    let field_syntax = ptr.to_node(&file);
    assert_eq!(field.syntax(), &*field_syntax);
}
