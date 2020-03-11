//! This module handles auto-magic editing actions applied together with users
//! edits. For example, if the user typed
//!
//! ```text
//!     foo
//!         .bar()
//!         .baz()
//!     |   // <- cursor is here
//! ```
//!
//! and types `.` next, we want to indent the dot.
//!
//! Language server executes such typing assists synchronously. That is, they
//! block user's typing and should be pretty fast for this reason!

mod on_enter;

use ra_db::{FilePosition, SourceDatabase};
use ra_fmt::leading_indent;
use ra_ide_db::RootDatabase;
use ra_syntax::{
    algo::find_node_at_offset,
    ast::{self, AstToken},
    AstNode, SourceFile, TextRange, TextUnit,
};
use ra_text_edit::TextEdit;

use crate::{source_change::SingleFileChange, SourceChange};

pub(crate) use on_enter::on_enter;

pub(crate) const TRIGGER_CHARS: &str = ".=>";

pub(crate) fn on_char_typed(
    db: &RootDatabase,
    position: FilePosition,
    char_typed: char,
) -> Option<SourceChange> {
    assert!(TRIGGER_CHARS.contains(char_typed));
    let file = &db.parse(position.file_id).tree();
    assert_eq!(file.syntax().text().char_at(position.offset), Some(char_typed));
    let single_file_change = on_char_typed_inner(file, position.offset, char_typed)?;
    Some(single_file_change.into_source_change(position.file_id))
}

fn on_char_typed_inner(
    file: &SourceFile,
    offset: TextUnit,
    char_typed: char,
) -> Option<SingleFileChange> {
    assert!(TRIGGER_CHARS.contains(char_typed));
    match char_typed {
        '.' => on_dot_typed(file, offset),
        '=' => on_eq_typed(file, offset),
        '>' => on_arrow_typed(file, offset),
        _ => unreachable!(),
    }
}

/// Returns an edit which should be applied after `=` was typed. Primarily,
/// this works when adding `let =`.
// FIXME: use a snippet completion instead of this hack here.
fn on_eq_typed(file: &SourceFile, offset: TextUnit) -> Option<SingleFileChange> {
    assert_eq!(file.syntax().text().char_at(offset), Some('='));
    let let_stmt: ast::LetStmt = find_node_at_offset(file.syntax(), offset)?;
    if let_stmt.has_semi() {
        return None;
    }
    if let Some(expr) = let_stmt.initializer() {
        let expr_range = expr.syntax().text_range();
        if expr_range.contains(offset) && offset != expr_range.start() {
            return None;
        }
        if file.syntax().text().slice(offset..expr_range.start()).contains_char('\n') {
            return None;
        }
    } else {
        return None;
    }
    let offset = let_stmt.syntax().text_range().end();
    Some(SingleFileChange {
        label: "add semicolon".to_string(),
        edit: TextEdit::insert(offset, ";".to_string()),
        cursor_position: None,
    })
}

/// Returns an edit which should be applied when a dot ('.') is typed on a blank line, indenting the line appropriately.
fn on_dot_typed(file: &SourceFile, offset: TextUnit) -> Option<SingleFileChange> {
    assert_eq!(file.syntax().text().char_at(offset), Some('.'));
    let whitespace =
        file.syntax().token_at_offset(offset).left_biased().and_then(ast::Whitespace::cast)?;

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

    Some(SingleFileChange {
        label: "reindent dot".to_string(),
        edit: TextEdit::replace(
            TextRange::from_to(offset - current_indent_len, offset),
            target_indent,
        ),
        cursor_position: Some(
            offset + target_indent_len - current_indent_len + TextUnit::of_char('.'),
        ),
    })
}

/// Adds a space after an arrow when `fn foo() { ... }` is turned into `fn foo() -> { ... }`
fn on_arrow_typed(file: &SourceFile, offset: TextUnit) -> Option<SingleFileChange> {
    let file_text = file.syntax().text();
    assert_eq!(file_text.char_at(offset), Some('>'));
    let after_arrow = offset + TextUnit::of_char('>');
    if file_text.char_at(after_arrow) != Some('{') {
        return None;
    }
    if find_node_at_offset::<ast::RetType>(file.syntax(), offset).is_none() {
        return None;
    }

    Some(SingleFileChange {
        label: "add space after return type".to_string(),
        edit: TextEdit::insert(after_arrow, " ".to_string()),
        cursor_position: Some(after_arrow),
    })
}

#[cfg(test)]
mod tests {
    use test_utils::{assert_eq_text, extract_offset};

    use super::*;

    fn do_type_char(char_typed: char, before: &str) -> Option<(String, SingleFileChange)> {
        let (offset, before) = extract_offset(before);
        let edit = TextEdit::insert(offset, char_typed.to_string());
        let before = edit.apply(&before);
        let parse = SourceFile::parse(&before);
        on_char_typed_inner(&parse.tree(), offset, char_typed)
            .map(|it| (it.edit.apply(&before), it))
    }

    fn type_char(char_typed: char, before: &str, after: &str) {
        let (actual, file_change) = do_type_char(char_typed, before)
            .unwrap_or_else(|| panic!("typing `{}` did nothing", char_typed));

        if after.contains("<|>") {
            let (offset, after) = extract_offset(after);
            assert_eq_text!(&after, &actual);
            assert_eq!(file_change.cursor_position, Some(offset))
        } else {
            assert_eq_text!(after, &actual);
        }
    }

    fn type_char_noop(char_typed: char, before: &str) {
        let file_change = do_type_char(char_typed, before);
        assert!(file_change.is_none())
    }

    #[test]
    fn test_on_eq_typed() {
        //     do_check(r"
        // fn foo() {
        //     let foo =<|>
        // }
        // ", r"
        // fn foo() {
        //     let foo =;
        // }
        // ");
        type_char(
            '=',
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

    #[test]
    fn indents_new_chain_call() {
        type_char(
            '.',
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
        type_char_noop(
            '.',
            r"
            pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
                self.child_impl(db, name)
                    <|>
            }
            ",
        )
    }

    #[test]
    fn indents_new_chain_call_with_semi() {
        type_char(
            '.',
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
        type_char_noop(
            '.',
            r"
            pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
                self.child_impl(db, name)
                    <|>;
            }
            ",
        )
    }

    #[test]
    fn indents_continued_chain_call() {
        type_char(
            '.',
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
        type_char_noop(
            '.',
            r"
            pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
                self.child_impl(db, name)
                    .first()
                    <|>
            }
            ",
        );
    }

    #[test]
    fn indents_middle_of_chain_call() {
        type_char(
            '.',
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
        type_char_noop(
            '.',
            r"
            fn source_impl() {
                let var = enum_defvariant_list().unwrap()
                    <|>
                    .nth(92)
                    .unwrap();
            }
            ",
        );
    }

    #[test]
    fn dont_indent_freestanding_dot() {
        type_char_noop(
            '.',
            r"
            pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
                <|>
            }
            ",
        );
        type_char_noop(
            '.',
            r"
            pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
            <|>
            }
            ",
        );
    }

    #[test]
    fn adds_space_after_return_type() {
        type_char('>', "fn foo() -<|>{ 92 }", "fn foo() -><|> { 92 }")
    }
}
