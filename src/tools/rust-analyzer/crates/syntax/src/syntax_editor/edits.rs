//! Structural editing for ast using `SyntaxEditor`

use crate::{
    AstToken, Direction, SyntaxElement, SyntaxKind, SyntaxNode, SyntaxToken, T,
    algo::neighbor,
    ast::{
        self, AstNode, Fn, GenericParam, HasGenericParams, HasName, edit::IndentLevel, make,
        syntax_factory::SyntaxFactory,
    },
    syntax_editor::{Position, SyntaxEditor},
};

impl SyntaxEditor {
    /// Adds a new generic param to the function using `SyntaxEditor`
    pub fn add_generic_param(&mut self, function: &Fn, new_param: GenericParam) {
        match function.generic_param_list() {
            Some(generic_param_list) => match generic_param_list.generic_params().last() {
                Some(last_param) => {
                    // There exists a generic param list and it's not empty
                    let position = generic_param_list.r_angle_token().map_or_else(
                        || Position::last_child_of(function.syntax()),
                        Position::before,
                    );

                    if last_param
                        .syntax()
                        .next_sibling_or_token()
                        .is_some_and(|it| it.kind() == SyntaxKind::COMMA)
                    {
                        self.insert(
                            Position::after(last_param.syntax()),
                            new_param.syntax().clone(),
                        );
                        self.insert(
                            Position::after(last_param.syntax()),
                            make::token(SyntaxKind::WHITESPACE),
                        );
                        self.insert(
                            Position::after(last_param.syntax()),
                            make::token(SyntaxKind::COMMA),
                        );
                    } else {
                        let elements = vec![
                            make::token(SyntaxKind::COMMA).into(),
                            make::token(SyntaxKind::WHITESPACE).into(),
                            new_param.syntax().clone().into(),
                        ];
                        self.insert_all(position, elements);
                    }
                }
                None => {
                    // There exists a generic param list but it's empty
                    let position = Position::after(generic_param_list.l_angle_token().unwrap());
                    self.insert(position, new_param.syntax());
                }
            },
            None => {
                // There was no generic param list
                let position = if let Some(name) = function.name() {
                    Position::after(name.syntax)
                } else if let Some(fn_token) = function.fn_token() {
                    Position::after(fn_token)
                } else if let Some(param_list) = function.param_list() {
                    Position::before(param_list.syntax)
                } else {
                    Position::last_child_of(function.syntax())
                };
                let elements = vec![
                    make::token(SyntaxKind::L_ANGLE).into(),
                    new_param.syntax().clone().into(),
                    make::token(SyntaxKind::R_ANGLE).into(),
                ];
                self.insert_all(position, elements);
            }
        }
    }
}

fn get_or_insert_comma_after(editor: &mut SyntaxEditor, syntax: &SyntaxNode) -> SyntaxToken {
    let make = SyntaxFactory::without_mappings();
    match syntax
        .siblings_with_tokens(Direction::Next)
        .filter_map(|it| it.into_token())
        .find(|it| it.kind() == T![,])
    {
        Some(it) => it,
        None => {
            let comma = make.token(T![,]);
            editor.insert(Position::after(syntax), &comma);
            comma
        }
    }
}

impl ast::AssocItemList {
    /// Adds a new associated item after all of the existing associated items.
    ///
    /// Attention! This function does align the first line of `item` with respect to `self`,
    /// but it does _not_ change indentation of other lines (if any).
    pub fn add_items(&self, editor: &mut SyntaxEditor, items: Vec<ast::AssocItem>) {
        let (indent, position, whitespace) = match self.assoc_items().last() {
            Some(last_item) => (
                IndentLevel::from_node(last_item.syntax()),
                Position::after(last_item.syntax()),
                "\n\n",
            ),
            None => match self.l_curly_token() {
                Some(l_curly) => {
                    normalize_ws_between_braces(editor, self.syntax());
                    (IndentLevel::from_token(&l_curly) + 1, Position::after(&l_curly), "\n")
                }
                None => (IndentLevel::single(), Position::last_child_of(self.syntax()), "\n"),
            },
        };

        let elements: Vec<SyntaxElement> = items
            .into_iter()
            .enumerate()
            .flat_map(|(i, item)| {
                let whitespace = if i != 0 { "\n\n" } else { whitespace };
                vec![
                    make::tokens::whitespace(&format!("{whitespace}{indent}")).into(),
                    item.syntax().clone().into(),
                ]
            })
            .collect();
        editor.insert_all(position, elements);
    }
}

impl ast::VariantList {
    pub fn add_variant(&self, editor: &mut SyntaxEditor, variant: &ast::Variant) {
        let make = SyntaxFactory::without_mappings();
        let (indent, position) = match self.variants().last() {
            Some(last_item) => (
                IndentLevel::from_node(last_item.syntax()),
                Position::after(get_or_insert_comma_after(editor, last_item.syntax())),
            ),
            None => match self.l_curly_token() {
                Some(l_curly) => {
                    normalize_ws_between_braces(editor, self.syntax());
                    (IndentLevel::from_token(&l_curly) + 1, Position::after(&l_curly))
                }
                None => (IndentLevel::single(), Position::last_child_of(self.syntax())),
            },
        };
        let elements: Vec<SyntaxElement> = vec![
            make.whitespace(&format!("{}{indent}", "\n")).into(),
            variant.syntax().clone().into(),
            make.token(T![,]).into(),
        ];
        editor.insert_all(position, elements);
    }
}

impl ast::Fn {
    pub fn replace_or_insert_body(&self, editor: &mut SyntaxEditor, body: ast::BlockExpr) {
        if let Some(old_body) = self.body() {
            editor.replace(old_body.syntax(), body.syntax());
        } else {
            let single_space = make::tokens::single_space();
            let elements = vec![single_space.into(), body.syntax().clone().into()];

            if let Some(semicolon) = self.semicolon_token() {
                editor.replace_with_many(semicolon, elements);
            } else {
                editor.insert_all(Position::last_child_of(self.syntax()), elements);
            }
        }
    }
}

fn normalize_ws_between_braces(editor: &mut SyntaxEditor, node: &SyntaxNode) -> Option<()> {
    let make = SyntaxFactory::without_mappings();
    let l = node
        .children_with_tokens()
        .filter_map(|it| it.into_token())
        .find(|it| it.kind() == T!['{'])?;
    let r = node
        .children_with_tokens()
        .filter_map(|it| it.into_token())
        .find(|it| it.kind() == T!['}'])?;

    let indent = IndentLevel::from_node(node);

    match l.next_sibling_or_token() {
        Some(ws) if ws.kind() == SyntaxKind::WHITESPACE => {
            if ws.next_sibling_or_token()?.into_token()? == r {
                editor.replace(ws, make.whitespace(&format!("\n{indent}")));
            }
        }
        Some(ws) if ws.kind() == T!['}'] => {
            editor.insert(Position::after(l), make.whitespace(&format!("\n{indent}")));
        }
        _ => (),
    }
    Some(())
}

pub trait Removable: AstNode {
    fn remove(&self, editor: &mut SyntaxEditor);
}

impl Removable for ast::TypeBoundList {
    fn remove(&self, editor: &mut SyntaxEditor) {
        match self.syntax().siblings_with_tokens(Direction::Prev).find(|it| it.kind() == T![:]) {
            Some(colon) => editor.delete_all(colon..=self.syntax().clone().into()),
            None => editor.delete(self.syntax()),
        }
    }
}

impl Removable for ast::Use {
    fn remove(&self, editor: &mut SyntaxEditor) {
        let make = SyntaxFactory::without_mappings();

        let next_ws = self
            .syntax()
            .next_sibling_or_token()
            .and_then(|it| it.into_token())
            .and_then(ast::Whitespace::cast);
        if let Some(next_ws) = next_ws {
            let ws_text = next_ws.syntax().text();
            if let Some(rest) = ws_text.strip_prefix('\n') {
                if rest.is_empty() {
                    editor.delete(next_ws.syntax());
                } else {
                    editor.replace(next_ws.syntax(), make.whitespace(rest));
                }
            }
        }

        editor.delete(self.syntax());
    }
}

impl Removable for ast::UseTree {
    fn remove(&self, editor: &mut SyntaxEditor) {
        for dir in [Direction::Next, Direction::Prev] {
            if let Some(next_use_tree) = neighbor(self, dir) {
                let separators = self
                    .syntax()
                    .siblings_with_tokens(dir)
                    .skip(1)
                    .take_while(|it| it.as_node() != Some(next_use_tree.syntax()));
                for sep in separators {
                    editor.delete(sep);
                }
                break;
            }
        }
        editor.delete(self.syntax());
    }
}

#[cfg(test)]
mod tests {
    use parser::Edition;
    use stdx::trim_indent;
    use test_utils::assert_eq_text;

    use crate::SourceFile;

    use super::*;

    fn ast_from_text<N: AstNode>(text: &str) -> N {
        let parse = SourceFile::parse(text, Edition::CURRENT);
        let node = match parse.tree().syntax().descendants().find_map(N::cast) {
            Some(it) => it,
            None => {
                let node = std::any::type_name::<N>();
                panic!("Failed to make ast node `{node}` from text {text}")
            }
        };
        let node = node.clone_subtree();
        assert_eq!(node.syntax().text_range().start(), 0.into());
        node
    }

    #[test]
    fn add_variant_to_empty_enum() {
        let make = SyntaxFactory::without_mappings();
        let variant = make.variant(None, make.name("Bar"), None, None);

        check_add_variant(
            r#"
enum Foo {}
"#,
            r#"
enum Foo {
    Bar,
}
"#,
            variant,
        );
    }

    #[test]
    fn add_variant_to_non_empty_enum() {
        let make = SyntaxFactory::without_mappings();
        let variant = make.variant(None, make.name("Baz"), None, None);

        check_add_variant(
            r#"
enum Foo {
    Bar,
}
"#,
            r#"
enum Foo {
    Bar,
    Baz,
}
"#,
            variant,
        );
    }

    #[test]
    fn add_variant_with_tuple_field_list() {
        let make = SyntaxFactory::without_mappings();
        let variant = make.variant(
            None,
            make.name("Baz"),
            Some(make.tuple_field_list([make.tuple_field(None, make.ty("bool"))]).into()),
            None,
        );

        check_add_variant(
            r#"
enum Foo {
    Bar,
}
"#,
            r#"
enum Foo {
    Bar,
    Baz(bool),
}
"#,
            variant,
        );
    }

    #[test]
    fn add_variant_with_record_field_list() {
        let make = SyntaxFactory::without_mappings();
        let variant = make.variant(
            None,
            make.name("Baz"),
            Some(
                make.record_field_list([make.record_field(None, make.name("x"), make.ty("bool"))])
                    .into(),
            ),
            None,
        );

        check_add_variant(
            r#"
enum Foo {
    Bar,
}
"#,
            r#"
enum Foo {
    Bar,
    Baz { x: bool },
}
"#,
            variant,
        );
    }

    fn check_add_variant(before: &str, expected: &str, variant: ast::Variant) {
        let enum_ = ast_from_text::<ast::Enum>(before);
        let mut editor = SyntaxEditor::new(enum_.syntax().clone());
        if let Some(it) = enum_.variant_list() {
            it.add_variant(&mut editor, &variant)
        }
        let edit = editor.finish();
        let after = edit.new_root.to_string();
        assert_eq_text!(&trim_indent(expected.trim()), &trim_indent(after.trim()));
    }
}
