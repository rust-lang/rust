//! Structural editing for ast using `SyntaxEditor`

use crate::{
    ast::{
        self, edit::IndentLevel, make, syntax_factory::SyntaxFactory, AstNode, Fn, GenericParam,
        HasGenericParams, HasName,
    },
    syntax_editor::{Position, SyntaxEditor},
    Direction, SyntaxElement, SyntaxKind, SyntaxNode, SyntaxToken, T,
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
