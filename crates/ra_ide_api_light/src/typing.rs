use ra_syntax::{
    AstNode, SourceFile, SyntaxKind::*,
    SyntaxNode, TextUnit, TextRange,
    algo::{find_node_at_offset, find_leaf_at_offset, LeafAtOffset},
    ast::{self, AstToken},
};
use ra_fmt::leading_indent;
use crate::{LocalEdit, TextEditBuilder};

pub fn on_enter(file: &SourceFile, offset: TextUnit) -> Option<LocalEdit> {
    let comment =
        find_leaf_at_offset(file.syntax(), offset).left_biased().and_then(ast::Comment::cast)?;

    if let ast::CommentFlavor::Multiline = comment.flavor() {
        return None;
    }

    let prefix = comment.prefix();
    if offset < comment.syntax().range().start() + TextUnit::of_str(prefix) + TextUnit::from(1) {
        return None;
    }

    let indent = node_indent(file, comment.syntax())?;
    let inserted = format!("\n{}{} ", indent, prefix);
    let cursor_position = offset + TextUnit::of_str(&inserted);
    let mut edit = TextEditBuilder::default();
    edit.insert(offset, inserted);
    Some(LocalEdit {
        label: "on enter".to_string(),
        edit: edit.finish(),
        cursor_position: Some(cursor_position),
    })
}

fn node_indent<'a>(file: &'a SourceFile, node: &SyntaxNode) -> Option<&'a str> {
    let ws = match find_leaf_at_offset(file.syntax(), node.range().start()) {
        LeafAtOffset::Between(l, r) => {
            assert!(r == node);
            l
        }
        LeafAtOffset::Single(n) => {
            assert!(n == node);
            return Some("");
        }
        LeafAtOffset::None => unreachable!(),
    };
    if ws.kind() != WHITESPACE {
        return None;
    }
    let text = ws.leaf_text().unwrap();
    let pos = text.as_str().rfind('\n').map(|it| it + 1).unwrap_or(0);
    Some(&text[pos..])
}

pub fn on_eq_typed(file: &SourceFile, eq_offset: TextUnit) -> Option<LocalEdit> {
    assert_eq!(file.syntax().text().char_at(eq_offset), Some('='));
    let let_stmt: &ast::LetStmt = find_node_at_offset(file.syntax(), eq_offset)?;
    if let_stmt.has_semi() {
        return None;
    }
    if let Some(expr) = let_stmt.initializer() {
        let expr_range = expr.syntax().range();
        if expr_range.contains(eq_offset) && eq_offset != expr_range.start() {
            return None;
        }
        if file.syntax().text().slice(eq_offset..expr_range.start()).contains('\n') {
            return None;
        }
    } else {
        return None;
    }
    let offset = let_stmt.syntax().range().end();
    let mut edit = TextEditBuilder::default();
    edit.insert(offset, ";".to_string());
    Some(LocalEdit {
        label: "add semicolon".to_string(),
        edit: edit.finish(),
        cursor_position: None,
    })
}

pub fn on_dot_typed(file: &SourceFile, dot_offset: TextUnit) -> Option<LocalEdit> {
    assert_eq!(file.syntax().text().char_at(dot_offset), Some('.'));

    let whitespace = find_leaf_at_offset(file.syntax(), dot_offset)
        .left_biased()
        .and_then(ast::Whitespace::cast)?;

    let current_indent = {
        let text = whitespace.text();
        let newline = text.rfind('\n')?;
        &text[newline + 1..]
    };
    let current_indent_len = TextUnit::of_str(current_indent);

    // Make sure dot is a part of call chain
    let field_expr = whitespace.syntax().parent().and_then(ast::FieldExpr::cast)?;
    let prev_indent = leading_indent(field_expr.syntax())?;
    let target_indent = format!("    {}", prev_indent);
    let target_indent_len = TextUnit::of_str(&target_indent);
    if current_indent_len == target_indent_len {
        return None;
    }
    let mut edit = TextEditBuilder::default();
    edit.replace(
        TextRange::from_to(dot_offset - current_indent_len, dot_offset),
        target_indent.into(),
    );
    let res = LocalEdit {
        label: "reindent dot".to_string(),
        edit: edit.finish(),
        cursor_position: Some(
            dot_offset + target_indent_len - current_indent_len + TextUnit::of_char('.'),
        ),
    };
    Some(res)
}

#[cfg(test)]
mod tests {
    use crate::test_utils::{add_cursor, assert_eq_text, extract_offset};

    use super::*;

    #[test]
    fn test_on_eq_typed() {
        fn type_eq(before: &str, after: &str) {
            let (offset, before) = extract_offset(before);
            let mut edit = TextEditBuilder::default();
            edit.insert(offset, "=".to_string());
            let before = edit.finish().apply(&before);
            let file = SourceFile::parse(&before);
            if let Some(result) = on_eq_typed(&file, offset) {
                let actual = result.edit.apply(&before);
                assert_eq_text!(after, &actual);
            } else {
                assert_eq_text!(&before, after)
            };
        }

        //     do_check(r"
        // fn foo() {
        //     let foo =<|>
        // }
        // ", r"
        // fn foo() {
        //     let foo =;
        // }
        // ");
        type_eq(
            r"
fn foo() {
    let foo <|> 1 + 1
}
",
            r"
fn foo() {
    let foo = 1 + 1;
}
",
        );
        //     do_check(r"
        // fn foo() {
        //     let foo =<|>
        //     let bar = 1;
        // }
        // ", r"
        // fn foo() {
        //     let foo =;
        //     let bar = 1;
        // }
        // ");
    }

    fn type_dot(before: &str, after: &str) {
        let (offset, before) = extract_offset(before);
        let mut edit = TextEditBuilder::default();
        edit.insert(offset, ".".to_string());
        let before = edit.finish().apply(&before);
        let file = SourceFile::parse(&before);
        if let Some(result) = on_dot_typed(&file, offset) {
            let actual = result.edit.apply(&before);
            assert_eq_text!(after, &actual);
        } else {
            assert_eq_text!(&before, after)
        };
    }

    #[test]
    fn indents_new_chain_call() {
        type_dot(
            r"
            pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
                self.child_impl(db, name)
                <|>
            }
            ",
            r"
            pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
                self.child_impl(db, name)
                    .
            }
            ",
        );
        type_dot(
            r"
            pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
                self.child_impl(db, name)
                    <|>
            }
            ",
            r"
            pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
                self.child_impl(db, name)
                    .
            }
            ",
        )
    }

    #[test]
    fn indents_new_chain_call_with_semi() {
        type_dot(
            r"
            pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
                self.child_impl(db, name)
                <|>;
            }
            ",
            r"
            pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
                self.child_impl(db, name)
                    .;
            }
            ",
        );
        type_dot(
            r"
            pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
                self.child_impl(db, name)
                    <|>;
            }
            ",
            r"
            pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
                self.child_impl(db, name)
                    .;
            }
            ",
        )
    }

    #[test]
    fn indents_continued_chain_call() {
        type_dot(
            r"
            pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
                self.child_impl(db, name)
                    .first()
                <|>
            }
            ",
            r"
            pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
                self.child_impl(db, name)
                    .first()
                    .
            }
            ",
        );
        type_dot(
            r"
            pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
                self.child_impl(db, name)
                    .first()
                    <|>
            }
            ",
            r"
            pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
                self.child_impl(db, name)
                    .first()
                    .
            }
            ",
        );
    }

    #[test]
    fn indents_middle_of_chain_call() {
        type_dot(
            r"
            fn source_impl() {
                let var = enum_defvariant_list().unwrap()
                <|>
                    .nth(92)
                    .unwrap();
            }
            ",
            r"
            fn source_impl() {
                let var = enum_defvariant_list().unwrap()
                    .
                    .nth(92)
                    .unwrap();
            }
            ",
        );
        type_dot(
            r"
            fn source_impl() {
                let var = enum_defvariant_list().unwrap()
                    <|>
                    .nth(92)
                    .unwrap();
            }
            ",
            r"
            fn source_impl() {
                let var = enum_defvariant_list().unwrap()
                    .
                    .nth(92)
                    .unwrap();
            }
            ",
        );
    }

    #[test]
    fn dont_indent_freestanding_dot() {
        type_dot(
            r"
            pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
                <|>
            }
            ",
            r"
            pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
                .
            }
            ",
        );
        type_dot(
            r"
            pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
            <|>
            }
            ",
            r"
            pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
            .
            }
            ",
        );
    }

    #[test]
    fn test_on_enter() {
        fn apply_on_enter(before: &str) -> Option<String> {
            let (offset, before) = extract_offset(before);
            let file = SourceFile::parse(&before);
            let result = on_enter(&file, offset)?;
            let actual = result.edit.apply(&before);
            let actual = add_cursor(&actual, result.cursor_position.unwrap());
            Some(actual)
        }

        fn do_check(before: &str, after: &str) {
            let actual = apply_on_enter(before).unwrap();
            assert_eq_text!(after, &actual);
        }

        fn do_check_noop(text: &str) {
            assert!(apply_on_enter(text).is_none())
        }

        do_check(
            r"
/// Some docs<|>
fn foo() {
}
",
            r"
/// Some docs
/// <|>
fn foo() {
}
",
        );
        do_check(
            r"
impl S {
    /// Some<|> docs.
    fn foo() {}
}
",
            r"
impl S {
    /// Some
    /// <|> docs.
    fn foo() {}
}
",
        );
        do_check_noop(r"<|>//! docz");
    }
}
