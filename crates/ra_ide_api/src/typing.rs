use ra_db::{FilePosition, SourceDatabase};
use ra_fmt::leading_indent;
use ra_syntax::{
    algo::find_node_at_offset,
    ast::{self, AstToken},
    AstNode, SmolStr, SourceFile,
    SyntaxKind::*,
    SyntaxToken, TextRange, TextUnit, TokenAtOffset,
};
use ra_text_edit::{TextEdit, TextEditBuilder};

use crate::{db::RootDatabase, SourceChange, SourceFileEdit};

pub(crate) fn on_enter(db: &RootDatabase, position: FilePosition) -> Option<SourceChange> {
    let parse = db.parse(position.file_id);
    let file = parse.tree();
    let comment = file
        .syntax()
        .token_at_offset(position.offset)
        .left_biased()
        .and_then(ast::Comment::cast)?;

    if comment.kind().shape.is_block() {
        return None;
    }

    let prefix = comment.prefix();
    if position.offset
        < comment.syntax().text_range().start() + TextUnit::of_str(prefix) + TextUnit::from(1)
    {
        return None;
    }

    let indent = node_indent(&file, comment.syntax())?;
    let inserted = format!("\n{}{} ", indent, prefix);
    let cursor_position = position.offset + TextUnit::of_str(&inserted);
    let mut edit = TextEditBuilder::default();
    edit.insert(position.offset, inserted);

    Some(
        SourceChange::source_file_edit(
            "on enter",
            SourceFileEdit { edit: edit.finish(), file_id: position.file_id },
        )
        .with_cursor(FilePosition { offset: cursor_position, file_id: position.file_id }),
    )
}

fn node_indent(file: &SourceFile, token: &SyntaxToken) -> Option<SmolStr> {
    let ws = match file.syntax().token_at_offset(token.text_range().start()) {
        TokenAtOffset::Between(l, r) => {
            assert!(r == *token);
            l
        }
        TokenAtOffset::Single(n) => {
            assert!(n == *token);
            return Some("".into());
        }
        TokenAtOffset::None => unreachable!(),
    };
    if ws.kind() != WHITESPACE {
        return None;
    }
    let text = ws.text();
    let pos = text.rfind('\n').map(|it| it + 1).unwrap_or(0);
    Some(text[pos..].into())
}

pub fn on_eq_typed(file: &SourceFile, eq_offset: TextUnit) -> Option<TextEdit> {
    assert_eq!(file.syntax().text().char_at(eq_offset), Some('='));
    let let_stmt: ast::LetStmt = find_node_at_offset(file.syntax(), eq_offset)?;
    if let_stmt.has_semi() {
        return None;
    }
    if let Some(expr) = let_stmt.initializer() {
        let expr_range = expr.syntax().text_range();
        if expr_range.contains(eq_offset) && eq_offset != expr_range.start() {
            return None;
        }
        if file.syntax().text().slice(eq_offset..expr_range.start()).contains_char('\n') {
            return None;
        }
    } else {
        return None;
    }
    let offset = let_stmt.syntax().text_range().end();
    let mut edit = TextEditBuilder::default();
    edit.insert(offset, ";".to_string());
    Some(edit.finish())
}

pub(crate) fn on_dot_typed(db: &RootDatabase, position: FilePosition) -> Option<SourceChange> {
    let parse = db.parse(position.file_id);
    assert_eq!(parse.tree().syntax().text().char_at(position.offset), Some('.'));

    let whitespace = parse
        .tree()
        .syntax()
        .token_at_offset(position.offset)
        .left_biased()
        .and_then(ast::Whitespace::cast)?;

    let current_indent = {
        let text = whitespace.text();
        let newline = text.rfind('\n')?;
        &text[newline + 1..]
    };
    let current_indent_len = TextUnit::of_str(current_indent);

    // Make sure dot is a part of call chain
    let field_expr = ast::FieldExpr::cast(whitespace.syntax().parent())?;
    let prev_indent = leading_indent(field_expr.syntax())?;
    let target_indent = format!("    {}", prev_indent);
    let target_indent_len = TextUnit::of_str(&target_indent);
    if current_indent_len == target_indent_len {
        return None;
    }
    let mut edit = TextEditBuilder::default();
    edit.replace(
        TextRange::from_to(position.offset - current_indent_len, position.offset),
        target_indent,
    );

    let res = SourceChange::source_file_edit_from("reindent dot", position.file_id, edit.finish())
        .with_cursor(FilePosition {
            offset: position.offset + target_indent_len - current_indent_len
                + TextUnit::of_char('.'),
            file_id: position.file_id,
        });

    Some(res)
}

#[cfg(test)]
mod tests {
    use test_utils::{add_cursor, assert_eq_text, extract_offset};

    use crate::mock_analysis::single_file;

    use super::*;

    #[test]
    fn test_on_eq_typed() {
        fn type_eq(before: &str, after: &str) {
            let (offset, before) = extract_offset(before);
            let mut edit = TextEditBuilder::default();
            edit.insert(offset, "=".to_string());
            let before = edit.finish().apply(&before);
            let parse = SourceFile::parse(&before);
            if let Some(result) = on_eq_typed(&parse.tree(), offset) {
                let actual = result.apply(&before);
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
        let (analysis, file_id) = single_file(&before);
        if let Some(result) = analysis.on_dot_typed(FilePosition { offset, file_id }).unwrap() {
            assert_eq!(result.source_file_edits.len(), 1);
            let actual = result.source_file_edits[0].edit.apply(&before);
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
            let (analysis, file_id) = single_file(&before);
            let result = analysis.on_enter(FilePosition { offset, file_id }).unwrap()?;

            assert_eq!(result.source_file_edits.len(), 1);
            let actual = result.source_file_edits[0].edit.apply(&before);
            let actual = add_cursor(&actual, result.cursor_position.unwrap().offset);
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
