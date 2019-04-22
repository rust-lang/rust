use std::iter;

use arrayvec::ArrayVec;
use ra_text_edit::TextEditBuilder;
use ra_syntax::{AstNode, TreeArc, ast, SyntaxKind::*, SyntaxElement, SourceFile, InsertPosition, Direction};
use ra_fmt::leading_indent;

pub struct AstEditor<N: AstNode> {
    original_ast: TreeArc<N>,
    ast: TreeArc<N>,
}

impl<N: AstNode> AstEditor<N> {
    pub fn new(node: &N) -> AstEditor<N> {
        AstEditor { original_ast: node.to_owned(), ast: node.to_owned() }
    }

    pub fn into_text_edit(self, builder: &mut TextEditBuilder) {
        // FIXME: compute a more fine-grained diff here.
        // If *you* know a nice algorithm to compute diff between two syntax
        // tree, tell me about it!
        builder.replace(self.original_ast.syntax().range(), self.ast().syntax().text().to_string());
    }

    pub fn ast(&self) -> &N {
        &*self.ast
    }

    fn insert_children<'a>(
        &self,
        position: InsertPosition<SyntaxElement<'_>>,
        to_insert: impl Iterator<Item = SyntaxElement<'a>>,
    ) -> TreeArc<N> {
        let new_syntax = self.ast().syntax().insert_children(position, to_insert);
        N::cast(&new_syntax).unwrap().to_owned()
    }
}

impl AstEditor<ast::NamedFieldList> {
    pub fn append_field(&mut self, field: &ast::NamedField) {
        self.insert_field(InsertPosition::Last, field)
    }

    pub fn make_multiline(&mut self) {
        let l_curly = match self.l_curly() {
            Some(it) => it,
            None => return,
        };
        let sibling = match l_curly.next_sibling_or_token() {
            Some(it) => it,
            None => return,
        };
        if sibling.as_token().map(|it| it.text().contains('\n')) == Some(true) {
            return;
        }

        let ws = tokens::WsBuilder::new(&format!(
            "\n{}",
            leading_indent(self.ast().syntax()).unwrap_or("")
        ));
        self.ast = self.insert_children(InsertPosition::After(l_curly), iter::once(ws.ws().into()));
    }

    pub fn insert_field(
        &mut self,
        position: InsertPosition<&'_ ast::NamedField>,
        field: &ast::NamedField,
    ) {
        let is_multiline = self.ast().syntax().text().contains('\n');
        let ws;
        let space = if is_multiline {
            ws = tokens::WsBuilder::new(&format!(
                "\n{}    ",
                leading_indent(self.ast().syntax()).unwrap_or("")
            ));
            ws.ws()
        } else {
            tokens::single_space()
        };

        let mut to_insert: ArrayVec<[SyntaxElement; 4]> = ArrayVec::new();
        to_insert.push(space.into());
        to_insert.push(field.syntax().into());
        to_insert.push(tokens::comma().into());

        macro_rules! after_l_curly {
            () => {{
                let anchor = match self.l_curly() {
                    Some(it) => it,
                    None => return,
                };
                InsertPosition::After(anchor)
            }};
        }

        macro_rules! after_field {
            ($anchor:expr) => {
                if let Some(comma) = $anchor
                    .syntax()
                    .siblings_with_tokens(Direction::Next)
                    .find(|it| it.kind() == COMMA)
                {
                    InsertPosition::After(comma)
                } else {
                    to_insert.insert(0, tokens::comma().into());
                    InsertPosition::After($anchor.syntax().into())
                }
            };
        };

        let position = match position {
            InsertPosition::First => after_l_curly!(),
            InsertPosition::Last => {
                if !is_multiline {
                    // don't insert comma before curly
                    to_insert.pop();
                }
                match self.ast().fields().last() {
                    Some(it) => after_field!(it),
                    None => after_l_curly!(),
                }
            }
            InsertPosition::Before(anchor) => InsertPosition::Before(anchor.syntax().into()),
            InsertPosition::After(anchor) => after_field!(anchor),
        };

        self.ast = self.insert_children(position, to_insert.iter().cloned());
    }

    fn l_curly(&self) -> Option<SyntaxElement> {
        self.ast().syntax().children_with_tokens().find(|it| it.kind() == L_CURLY)
    }
}

pub struct AstBuilder<N: AstNode> {
    _phantom: std::marker::PhantomData<N>,
}

impl AstBuilder<ast::NamedField> {
    fn from_text(text: &str) -> TreeArc<ast::NamedField> {
        ast_node_from_file_text(&format!("fn f() {{ S {{ {}, }} }}", text))
    }

    pub fn from_pieces(name: &ast::NameRef, expr: Option<&ast::Expr>) -> TreeArc<ast::NamedField> {
        match expr {
            Some(expr) => Self::from_text(&format!("{}: {}", name.syntax(), expr.syntax())),
            None => Self::from_text(&name.syntax().to_string()),
        }
    }
}

impl AstBuilder<ast::Expr> {
    fn from_text(text: &str) -> TreeArc<ast::Expr> {
        ast_node_from_file_text(&format!("fn f() {{ {}; }}", text))
    }

    pub fn unit() -> TreeArc<ast::Expr> {
        Self::from_text("()")
    }
}

impl AstBuilder<ast::NameRef> {
    pub fn new(text: &str) -> TreeArc<ast::NameRef> {
        ast_node_from_file_text(&format!("fn f() {{ {}; }}", text))
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
        static ref SOURCE_FILE: TreeArc<SourceFile> = SourceFile::parse(",\n; ;");
    }

    pub(crate) fn comma() -> SyntaxToken<'static> {
        SOURCE_FILE
            .syntax()
            .descendants_with_tokens()
            .filter_map(|it| it.as_token())
            .find(|it| it.kind() == COMMA)
            .unwrap()
    }

    pub(crate) fn single_space() -> SyntaxToken<'static> {
        SOURCE_FILE
            .syntax()
            .descendants_with_tokens()
            .filter_map(|it| it.as_token())
            .find(|it| it.kind() == WHITESPACE && it.text().as_str() == " ")
            .unwrap()
    }

    pub(crate) struct WsBuilder(TreeArc<SourceFile>);

    impl WsBuilder {
        pub(crate) fn new(text: &str) -> WsBuilder {
            WsBuilder(SourceFile::parse(text))
        }
        pub(crate) fn ws(&self) -> SyntaxToken<'_> {
            self.0.syntax().first_child_or_token().unwrap().as_token().unwrap()
        }
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
