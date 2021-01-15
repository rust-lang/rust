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

use ide_db::{
    base_db::{FilePosition, SourceDatabase},
    RootDatabase,
};
use syntax::{
    algo::find_node_at_offset,
    ast::{self, edit::IndentLevel, AstToken},
    AstNode, SourceFile,
    SyntaxKind::{FIELD_EXPR, METHOD_CALL_EXPR},
    TextRange, TextSize,
};

use text_edit::TextEdit;

use crate::SourceChange;

pub(crate) use on_enter::on_enter;

pub(crate) const TRIGGER_CHARS: &str = ".=>";

// Feature: On Typing Assists
//
// Some features trigger on typing certain characters:
//
// - typing `let =` tries to smartly add `;` if `=` is followed by an existing expression
// - typing `.` in a chain method call auto-indents
//
// VS Code::
//
// Add the following to `settings.json`:
// [source,json]
// ----
// "editor.formatOnType": true,
// ----
pub(crate) fn on_char_typed(
    db: &RootDatabase,
    position: FilePosition,
    char_typed: char,
) -> Option<SourceChange> {
    assert!(TRIGGER_CHARS.contains(char_typed));
    let file = &db.parse(position.file_id).tree();
    assert_eq!(file.syntax().text().char_at(position.offset), Some(char_typed));
    let edit = on_char_typed_inner(file, position.offset, char_typed)?;
    Some(SourceChange::from_text_edit(position.file_id, edit))
}

fn on_char_typed_inner(file: &SourceFile, offset: TextSize, char_typed: char) -> Option<TextEdit> {
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
fn on_eq_typed(file: &SourceFile, offset: TextSize) -> Option<TextEdit> {
    assert_eq!(file.syntax().text().char_at(offset), Some('='));
    let let_stmt: ast::LetStmt = find_node_at_offset(file.syntax(), offset)?;
    if let_stmt.semicolon_token().is_some() {
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
    Some(TextEdit::insert(offset, ";".to_string()))
}

/// Returns an edit which should be applied when a dot ('.') is typed on a blank line, indenting the line appropriately.
fn on_dot_typed(file: &SourceFile, offset: TextSize) -> Option<TextEdit> {
    assert_eq!(file.syntax().text().char_at(offset), Some('.'));
    let whitespace =
        file.syntax().token_at_offset(offset).left_biased().and_then(ast::Whitespace::cast)?;

    let current_indent = {
        let text = whitespace.text();
        let newline = text.rfind('\n')?;
        &text[newline + 1..]
    };
    let current_indent_len = TextSize::of(current_indent);

    let parent = whitespace.syntax().parent();
    // Make sure dot is a part of call chain
    if !matches!(parent.kind(), FIELD_EXPR | METHOD_CALL_EXPR) {
        return None;
    }
    let prev_indent = IndentLevel::from_node(&parent);
    let target_indent = format!("    {}", prev_indent);
    let target_indent_len = TextSize::of(&target_indent);
    if current_indent_len == target_indent_len {
        return None;
    }

    Some(TextEdit::replace(TextRange::new(offset - current_indent_len, offset), target_indent))
}

/// Adds a space after an arrow when `fn foo() { ... }` is turned into `fn foo() -> { ... }`
fn on_arrow_typed(file: &SourceFile, offset: TextSize) -> Option<TextEdit> {
    let file_text = file.syntax().text();
    assert_eq!(file_text.char_at(offset), Some('>'));
    let after_arrow = offset + TextSize::of('>');
    if file_text.char_at(after_arrow) != Some('{') {
        return None;
    }
    if find_node_at_offset::<ast::RetType>(file.syntax(), offset).is_none() {
        return None;
    }

    Some(TextEdit::insert(after_arrow, " ".to_string()))
}

#[cfg(test)]
mod tests {
    use test_utils::{assert_eq_text, extract_offset};

    use super::*;

    fn do_type_char(char_typed: char, before: &str) -> Option<String> {
        let (offset, before) = extract_offset(before);
        let edit = TextEdit::insert(offset, char_typed.to_string());
        let mut before = before.to_string();
        edit.apply(&mut before);
        let parse = SourceFile::parse(&before);
        on_char_typed_inner(&parse.tree(), offset, char_typed).map(|it| {
            it.apply(&mut before);
            before.to_string()
        })
    }

    fn type_char(char_typed: char, ra_fixture_before: &str, ra_fixture_after: &str) {
        let actual = do_type_char(char_typed, ra_fixture_before)
            .unwrap_or_else(|| panic!("typing `{}` did nothing", char_typed));

        assert_eq_text!(ra_fixture_after, &actual);
    }

    fn type_char_noop(char_typed: char, before: &str) {
        let file_change = do_type_char(char_typed, before);
        assert!(file_change.is_none())
    }

    #[test]
    fn test_on_eq_typed() {
        //     do_check(r"
        // fn foo() {
        //     let foo =$0
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
    let foo $0 1 + 1
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
        //     let foo =$0
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
            fn main() {
                xs.foo()
                $0
            }
            ",
            r"
            fn main() {
                xs.foo()
                    .
            }
            ",
        );
        type_char_noop(
            '.',
            r"
            fn main() {
                xs.foo()
                    $0
            }
            ",
        )
    }

    #[test]
    fn indents_new_chain_call_with_semi() {
        type_char(
            '.',
            r"
            fn main() {
                xs.foo()
                $0;
            }
            ",
            r"
            fn main() {
                xs.foo()
                    .;
            }
            ",
        );
        type_char_noop(
            '.',
            r"
            fn main() {
                xs.foo()
                    $0;
            }
            ",
        )
    }

    #[test]
    fn indents_new_chain_call_with_let() {
        type_char(
            '.',
            r#"
fn main() {
    let _ = foo
    $0
    bar()
}
"#,
            r#"
fn main() {
    let _ = foo
        .
    bar()
}
"#,
        );
    }

    #[test]
    fn indents_continued_chain_call() {
        type_char(
            '.',
            r"
            fn main() {
                xs.foo()
                    .first()
                $0
            }
            ",
            r"
            fn main() {
                xs.foo()
                    .first()
                    .
            }
            ",
        );
        type_char_noop(
            '.',
            r"
            fn main() {
                xs.foo()
                    .first()
                    $0
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
                $0
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
                    $0
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
            fn main() {
                $0
            }
            ",
        );
        type_char_noop(
            '.',
            r"
            fn main() {
            $0
            }
            ",
        );
    }

    #[test]
    fn adds_space_after_return_type() {
        type_char('>', "fn foo() -$0{ 92 }", "fn foo() -> { 92 }")
    }
}
