use arrayvec::ArrayVec;
use ra_text_edit::{TextEdit, TextEditBuilder};
use ra_syntax::{AstNode, TreeArc, ast, SyntaxKind::*, SyntaxElement, SourceFile, InsertPosition, Direction};

pub struct AstEditor<N: AstNode> {
    original_ast: TreeArc<N>,
    ast: TreeArc<N>,
}

impl<N: AstNode> AstEditor<N> {
    pub fn new(node: &N) -> AstEditor<N> {
        AstEditor { original_ast: node.to_owned(), ast: node.to_owned() }
    }

    pub fn into_text_edit(self) -> TextEdit {
        // FIXME: compute a more fine-grained diff here.
        // If *you* know a nice algorithm to compute diff between two syntax
        // tree, tell me about it!
        let mut builder = TextEditBuilder::default();
        builder.replace(self.original_ast.syntax().range(), self.ast().syntax().text().to_string());
        builder.finish()
    }

    pub fn ast(&self) -> &N {
        &*self.ast
    }
}

impl AstEditor<ast::NamedFieldList> {
    pub fn append_field(&mut self, field: &ast::NamedField) {
        self.insert_field(InsertPosition::Last, field)
    }

    pub fn insert_field(
        &mut self,
        position: InsertPosition<&'_ ast::NamedField>,
        field: &ast::NamedField,
    ) {
        let mut to_insert: ArrayVec<[SyntaxElement; 2]> =
            [field.syntax().into(), tokens::comma().into()].into();
        let position = match position {
            InsertPosition::First => {
                let anchor = match self
                    .ast()
                    .syntax()
                    .children_with_tokens()
                    .find(|it| it.kind() == L_CURLY)
                {
                    Some(it) => it,
                    None => return,
                };
                InsertPosition::After(anchor)
            }
            InsertPosition::Last => {
                let anchor = match self
                    .ast()
                    .syntax()
                    .children_with_tokens()
                    .find(|it| it.kind() == R_CURLY)
                {
                    Some(it) => it,
                    None => return,
                };
                InsertPosition::Before(anchor)
            }
            InsertPosition::Before(anchor) => InsertPosition::Before(anchor.syntax().into()),
            InsertPosition::After(anchor) => {
                if let Some(comma) = anchor
                    .syntax()
                    .siblings_with_tokens(Direction::Next)
                    .find(|it| it.kind() == COMMA)
                {
                    InsertPosition::After(comma)
                } else {
                    to_insert.insert(0, tokens::comma().into());
                    InsertPosition::After(anchor.syntax().into())
                }
            }
        };
        self.ast = insert_children_into_ast(self.ast(), position, to_insert.iter().cloned());
    }
}

fn insert_children_into_ast<'a, N: AstNode>(
    node: &N,
    position: InsertPosition<SyntaxElement<'_>>,
    to_insert: impl Iterator<Item = SyntaxElement<'a>>,
) -> TreeArc<N> {
    let new_syntax = node.syntax().insert_children(position, to_insert);
    N::cast(&new_syntax).unwrap().to_owned()
}

pub struct AstBuilder<N: AstNode> {
    _phantom: std::marker::PhantomData<N>,
}

impl AstBuilder<ast::NamedField> {
    pub fn from_text(text: &str) -> TreeArc<ast::NamedField> {
        ast_node_from_file_text(&format!("fn f() {{ S {{ {}, }} }}", text))
    }
}

fn ast_node_from_file_text<N: AstNode>(text: &str) -> TreeArc<N> {
    let file = SourceFile::parse(text);
    let res = file.syntax().descendants().find_map(N::cast).unwrap().to_owned();
    res
}

mod tokens {
    use lazy_static::lazy_static;
    use ra_syntax::{AstNode, SourceFile, TreeArc, SyntaxToken, SyntaxKind::*};

    lazy_static! {
        static ref SOURCE_FILE: TreeArc<SourceFile> = SourceFile::parse(",");
    }

    pub(crate) fn comma() -> SyntaxToken<'static> {
        SOURCE_FILE
            .syntax()
            .descendants_with_tokens()
            .filter_map(|it| it.as_token())
            .find(|it| it.kind() == COMMA)
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ra_syntax::SourceFile;

    #[test]
    fn structure_editing() {
        let file = SourceFile::parse(
            "\
fn foo() {
    let s = S {
        original: 92,
    }
}
",
        );
        let field_list = file.syntax().descendants().find_map(ast::NamedFieldList::cast).unwrap();
        let mut editor = AstEditor::new(field_list);

        let field = AstBuilder::<ast::NamedField>::from_text("first_inserted: 1");
        editor.append_field(&field);
        let field = AstBuilder::<ast::NamedField>::from_text("second_inserted: 2");
        editor.append_field(&field);
        eprintln!("{}", editor.ast().syntax());
    }
}
