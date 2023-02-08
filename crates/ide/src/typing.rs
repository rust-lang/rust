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
    algo::{ancestors_at_offset, find_node_at_offset},
    ast::{self, edit::IndentLevel, AstToken},
    AstNode, Parse, SourceFile, SyntaxKind, TextRange, TextSize, T,
};

use text_edit::{Indel, TextEdit};

use crate::SourceChange;

pub(crate) use on_enter::on_enter;

// Don't forget to add new trigger characters to `server_capabilities` in `caps.rs`.
pub(crate) const TRIGGER_CHARS: &str = ".=<>{";

struct ExtendedTextEdit {
    edit: TextEdit,
    is_snippet: bool,
}

// Feature: On Typing Assists
//
// Some features trigger on typing certain characters:
//
// - typing `let =` tries to smartly add `;` if `=` is followed by an existing expression
// - typing `=` between two expressions adds `;` when in statement position
// - typing `=` to turn an assignment into an equality comparison removes `;` when in expression position
// - typing `.` in a chain method call auto-indents
// - typing `{` in front of an expression inserts a closing `}` after the expression
// - typing `{` in a use item adds a closing `}` in the right place
//
// VS Code::
//
// Add the following to `settings.json`:
// [source,json]
// ----
// "editor.formatOnType": true,
// ----
//
// image::https://user-images.githubusercontent.com/48062697/113166163-69758500-923a-11eb-81ee-eb33ec380399.gif[]
// image::https://user-images.githubusercontent.com/48062697/113171066-105c2000-923f-11eb-87ab-f4a263346567.gif[]
pub(crate) fn on_char_typed(
    db: &RootDatabase,
    position: FilePosition,
    char_typed: char,
) -> Option<SourceChange> {
    if !stdx::always!(TRIGGER_CHARS.contains(char_typed)) {
        return None;
    }
    let file = &db.parse(position.file_id);
    if !stdx::always!(file.tree().syntax().text().char_at(position.offset) == Some(char_typed)) {
        return None;
    }
    let edit = on_char_typed_inner(file, position.offset, char_typed)?;
    let mut sc = SourceChange::from_text_edit(position.file_id, edit.edit);
    sc.is_snippet = edit.is_snippet;
    Some(sc)
}

fn on_char_typed_inner(
    file: &Parse<SourceFile>,
    offset: TextSize,
    char_typed: char,
) -> Option<ExtendedTextEdit> {
    if !stdx::always!(TRIGGER_CHARS.contains(char_typed)) {
        return None;
    }
    return match char_typed {
        '.' => conv(on_dot_typed(&file.tree(), offset)),
        '=' => conv(on_eq_typed(&file.tree(), offset)),
        '<' => on_left_angle_typed(&file.tree(), offset),
        '>' => conv(on_right_angle_typed(&file.tree(), offset)),
        '{' => conv(on_opening_brace_typed(file, offset)),
        _ => return None,
    };

    fn conv(text_edit: Option<TextEdit>) -> Option<ExtendedTextEdit> {
        Some(ExtendedTextEdit { edit: text_edit?, is_snippet: false })
    }
}

/// Inserts a closing `}` when the user types an opening `{`, wrapping an existing expression in a
/// block, or a part of a `use` item.
fn on_opening_brace_typed(file: &Parse<SourceFile>, offset: TextSize) -> Option<TextEdit> {
    if !stdx::always!(file.tree().syntax().text().char_at(offset) == Some('{')) {
        return None;
    }

    let brace_token = file.tree().syntax().token_at_offset(offset).right_biased()?;
    if brace_token.kind() != SyntaxKind::L_CURLY {
        return None;
    }

    // Remove the `{` to get a better parse tree, and reparse.
    let range = brace_token.text_range();
    if !stdx::always!(range.len() == TextSize::of('{')) {
        return None;
    }
    let file = file.reparse(&Indel::delete(range));

    if let Some(edit) = brace_expr(&file.tree(), offset) {
        return Some(edit);
    }

    if let Some(edit) = brace_use_path(&file.tree(), offset) {
        return Some(edit);
    }

    return None;

    fn brace_use_path(file: &SourceFile, offset: TextSize) -> Option<TextEdit> {
        let segment: ast::PathSegment = find_node_at_offset(file.syntax(), offset)?;
        if segment.syntax().text_range().start() != offset {
            return None;
        }

        let tree: ast::UseTree = find_node_at_offset(file.syntax(), offset)?;

        Some(TextEdit::insert(
            tree.syntax().text_range().end() + TextSize::of("{"),
            "}".to_string(),
        ))
    }

    fn brace_expr(file: &SourceFile, offset: TextSize) -> Option<TextEdit> {
        let mut expr: ast::Expr = find_node_at_offset(file.syntax(), offset)?;
        if expr.syntax().text_range().start() != offset {
            return None;
        }

        // Enclose the outermost expression starting at `offset`
        while let Some(parent) = expr.syntax().parent() {
            if parent.text_range().start() != expr.syntax().text_range().start() {
                break;
            }

            match ast::Expr::cast(parent) {
                Some(parent) => expr = parent,
                None => break,
            }
        }

        // If it's a statement in a block, we don't know how many statements should be included
        if ast::ExprStmt::can_cast(expr.syntax().parent()?.kind()) {
            return None;
        }

        // Insert `}` right after the expression.
        Some(TextEdit::insert(
            expr.syntax().text_range().end() + TextSize::of("{"),
            "}".to_string(),
        ))
    }
}

/// Returns an edit which should be applied after `=` was typed. Primarily,
/// this works when adding `let =`.
// FIXME: use a snippet completion instead of this hack here.
fn on_eq_typed(file: &SourceFile, offset: TextSize) -> Option<TextEdit> {
    if !stdx::always!(file.syntax().text().char_at(offset) == Some('=')) {
        return None;
    }

    if let Some(edit) = let_stmt(file, offset) {
        return Some(edit);
    }
    if let Some(edit) = assign_expr(file, offset) {
        return Some(edit);
    }
    if let Some(edit) = assign_to_eq(file, offset) {
        return Some(edit);
    }

    return None;

    fn assign_expr(file: &SourceFile, offset: TextSize) -> Option<TextEdit> {
        let binop: ast::BinExpr = find_node_at_offset(file.syntax(), offset)?;
        if !matches!(binop.op_kind(), Some(ast::BinaryOp::Assignment { op: None })) {
            return None;
        }

        // Parent must be `ExprStmt` or `StmtList` for `;` to be valid.
        if let Some(expr_stmt) = ast::ExprStmt::cast(binop.syntax().parent()?) {
            if expr_stmt.semicolon_token().is_some() {
                return None;
            }
        } else if !ast::StmtList::can_cast(binop.syntax().parent()?.kind()) {
            return None;
        }

        let expr = binop.rhs()?;
        let expr_range = expr.syntax().text_range();
        if expr_range.contains(offset) && offset != expr_range.start() {
            return None;
        }
        if file.syntax().text().slice(offset..expr_range.start()).contains_char('\n') {
            return None;
        }
        let offset = expr.syntax().text_range().end();
        Some(TextEdit::insert(offset, ";".to_string()))
    }

    /// `a =$0 b;` removes the semicolon if an expression is valid in this context.
    fn assign_to_eq(file: &SourceFile, offset: TextSize) -> Option<TextEdit> {
        let binop: ast::BinExpr = find_node_at_offset(file.syntax(), offset)?;
        if !matches!(binop.op_kind(), Some(ast::BinaryOp::CmpOp(ast::CmpOp::Eq { negated: false })))
        {
            return None;
        }

        let expr_stmt = ast::ExprStmt::cast(binop.syntax().parent()?)?;
        let semi = expr_stmt.semicolon_token()?;

        if expr_stmt.syntax().next_sibling().is_some() {
            // Not the last statement in the list.
            return None;
        }

        Some(TextEdit::delete(semi.text_range()))
    }

    fn let_stmt(file: &SourceFile, offset: TextSize) -> Option<TextEdit> {
        let let_stmt: ast::LetStmt = find_node_at_offset(file.syntax(), offset)?;
        if let_stmt.semicolon_token().is_some() {
            return None;
        }
        let expr = let_stmt.initializer()?;
        let expr_range = expr.syntax().text_range();
        if expr_range.contains(offset) && offset != expr_range.start() {
            return None;
        }
        if file.syntax().text().slice(offset..expr_range.start()).contains_char('\n') {
            return None;
        }
        // Good indicator that we will insert into a bad spot, so bail out.
        if expr.syntax().descendants().any(|it| it.kind() == SyntaxKind::ERROR) {
            return None;
        }
        let offset = let_stmt.syntax().text_range().end();
        Some(TextEdit::insert(offset, ";".to_string()))
    }
}

/// Returns an edit which should be applied when a dot ('.') is typed on a blank line, indenting the line appropriately.
fn on_dot_typed(file: &SourceFile, offset: TextSize) -> Option<TextEdit> {
    if !stdx::always!(file.syntax().text().char_at(offset) == Some('.')) {
        return None;
    }
    let whitespace =
        file.syntax().token_at_offset(offset).left_biased().and_then(ast::Whitespace::cast)?;

    // if prior is fn call over multiple lines dont indent
    // or if previous is method call over multiples lines keep that indent
    let current_indent = {
        let text = whitespace.text();
        let (_prefix, suffix) = text.rsplit_once('\n')?;
        suffix
    };
    let current_indent_len = TextSize::of(current_indent);

    let parent = whitespace.syntax().parent()?;
    // Make sure dot is a part of call chain
    let receiver = if let Some(field_expr) = ast::FieldExpr::cast(parent.clone()) {
        field_expr.expr()?
    } else if let Some(method_call_expr) = ast::MethodCallExpr::cast(parent.clone()) {
        method_call_expr.receiver()?
    } else {
        return None;
    };

    let receiver_is_multiline = receiver.syntax().text().find_char('\n').is_some();
    let target_indent = match (receiver, receiver_is_multiline) {
        // if receiver is multiline field or method call, just take the previous `.` indentation
        (ast::Expr::MethodCallExpr(expr), true) => {
            expr.dot_token().as_ref().map(IndentLevel::from_token)
        }
        (ast::Expr::FieldExpr(expr), true) => {
            expr.dot_token().as_ref().map(IndentLevel::from_token)
        }
        // if receiver is multiline expression, just keeps its indentation
        (_, true) => Some(IndentLevel::from_node(&parent)),
        _ => None,
    };
    let target_indent = match target_indent {
        Some(x) => x,
        // in all other cases, take previous indentation and indent once
        None => IndentLevel::from_node(&parent) + 1,
    }
    .to_string();

    if current_indent_len == TextSize::of(&target_indent) {
        return None;
    }

    Some(TextEdit::replace(TextRange::new(offset - current_indent_len, offset), target_indent))
}

/// Add closing `>` for generic arguments/parameters.
fn on_left_angle_typed(file: &SourceFile, offset: TextSize) -> Option<ExtendedTextEdit> {
    let file_text = file.syntax().text();
    if !stdx::always!(file_text.char_at(offset) == Some('<')) {
        return None;
    }

    // Find the next non-whitespace char in the line.
    let mut next_offset = offset + TextSize::of('<');
    while file_text.char_at(next_offset) == Some(' ') {
        next_offset += TextSize::of(' ')
    }
    if file_text.char_at(next_offset) == Some('>') {
        return None;
    }

    let range = TextRange::at(offset, TextSize::of('<'));
    if let Some(t) = file.syntax().token_at_offset(offset).left_biased() {
        if T![impl] == t.kind() {
            return Some(ExtendedTextEdit {
                edit: TextEdit::replace(range, "<$0>".to_string()),
                is_snippet: true,
            });
        }
    }

    if ancestors_at_offset(file.syntax(), offset)
        .find(|n| {
            ast::GenericParamList::can_cast(n.kind()) || ast::GenericArgList::can_cast(n.kind())
        })
        .is_some()
    {
        return Some(ExtendedTextEdit {
            edit: TextEdit::replace(range, "<$0>".to_string()),
            is_snippet: true,
        });
    }

    None
}

/// Adds a space after an arrow when `fn foo() { ... }` is turned into `fn foo() -> { ... }`
fn on_right_angle_typed(file: &SourceFile, offset: TextSize) -> Option<TextEdit> {
    let file_text = file.syntax().text();
    if !stdx::always!(file_text.char_at(offset) == Some('>')) {
        return None;
    }
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

    impl ExtendedTextEdit {
        fn apply(&self, text: &mut String) {
            self.edit.apply(text);
        }
    }

    fn do_type_char(char_typed: char, before: &str) -> Option<String> {
        let (offset, mut before) = extract_offset(before);
        let edit = TextEdit::insert(offset, char_typed.to_string());
        edit.apply(&mut before);
        let parse = SourceFile::parse(&before);
        on_char_typed_inner(&parse, offset, char_typed).map(|it| {
            it.apply(&mut before);
            before.to_string()
        })
    }

    fn type_char(char_typed: char, ra_fixture_before: &str, ra_fixture_after: &str) {
        let actual = do_type_char(char_typed, ra_fixture_before)
            .unwrap_or_else(|| panic!("typing `{char_typed}` did nothing"));

        assert_eq_text!(ra_fixture_after, &actual);
    }

    fn type_char_noop(char_typed: char, ra_fixture_before: &str) {
        let file_change = do_type_char(char_typed, ra_fixture_before);
        assert!(file_change.is_none())
    }

    #[test]
    fn test_semi_after_let() {
        type_char_noop(
            '=',
            r"
fn foo() {
    let foo =$0
}
",
        );
        type_char(
            '=',
            r#"
fn foo() {
    let foo $0 1 + 1
}
"#,
            r#"
fn foo() {
    let foo = 1 + 1;
}
"#,
        );
        type_char_noop(
            '=',
            r#"
fn foo() {
    let difference $0(counts: &HashMap<(char, char), u64>, last: char) -> u64 {
        // ...
    }
}
"#,
        );
        type_char_noop(
            '=',
            r"
fn foo() {
    let foo =$0
    let bar = 1;
}
",
        );
    }

    #[test]
    fn test_semi_after_assign() {
        type_char(
            '=',
            r#"
fn f() {
    i $0 0
}
"#,
            r#"
fn f() {
    i = 0;
}
"#,
        );
        type_char(
            '=',
            r#"
fn f() {
    i $0 0
    i
}
"#,
            r#"
fn f() {
    i = 0;
    i
}
"#,
        );
        type_char_noop(
            '=',
            r#"
fn f(x: u8) {
    if x $0
}
"#,
        );
        type_char_noop(
            '=',
            r#"
fn f(x: u8) {
    if x $0 {}
}
"#,
        );
        type_char_noop(
            '=',
            r#"
fn f(x: u8) {
    if x $0 0 {}
}
"#,
        );
        type_char_noop(
            '=',
            r#"
fn f() {
    g(i $0 0);
}
"#,
        );
    }

    #[test]
    fn assign_to_eq() {
        type_char(
            '=',
            r#"
fn f(a: u8) {
    a =$0 0;
}
"#,
            r#"
fn f(a: u8) {
    a == 0
}
"#,
        );
        type_char(
            '=',
            r#"
fn f(a: u8) {
    a $0= 0;
}
"#,
            r#"
fn f(a: u8) {
    a == 0
}
"#,
        );
        type_char_noop(
            '=',
            r#"
fn f(a: u8) {
    let e = a =$0 0;
}
"#,
        );
        type_char_noop(
            '=',
            r#"
fn f(a: u8) {
    let e = a =$0 0;
    e
}
"#,
        );
    }

    #[test]
    fn indents_new_chain_call() {
        type_char(
            '.',
            r#"
fn main() {
    xs.foo()
    $0
}
            "#,
            r#"
fn main() {
    xs.foo()
        .
}
            "#,
        );
        type_char_noop(
            '.',
            r#"
fn main() {
    xs.foo()
        $0
}
            "#,
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
            r#"
fn main() {
    xs.foo()
        .;
}
            "#,
        );
        type_char_noop(
            '.',
            r#"
fn main() {
    xs.foo()
        $0;
}
            "#,
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
            r#"
fn main() {
    xs.foo()
        .first()
    $0
}
            "#,
            r#"
fn main() {
    xs.foo()
        .first()
        .
}
            "#,
        );
        type_char_noop(
            '.',
            r#"
fn main() {
    xs.foo()
        .first()
        $0
}
            "#,
        );
    }

    #[test]
    fn indents_middle_of_chain_call() {
        type_char(
            '.',
            r#"
fn source_impl() {
    let var = enum_defvariant_list().unwrap()
    $0
        .nth(92)
        .unwrap();
}
            "#,
            r#"
fn source_impl() {
    let var = enum_defvariant_list().unwrap()
        .
        .nth(92)
        .unwrap();
}
            "#,
        );
        type_char_noop(
            '.',
            r#"
fn source_impl() {
    let var = enum_defvariant_list().unwrap()
        $0
        .nth(92)
        .unwrap();
}
            "#,
        );
    }

    #[test]
    fn dont_indent_freestanding_dot() {
        type_char_noop(
            '.',
            r#"
fn main() {
    $0
}
            "#,
        );
        type_char_noop(
            '.',
            r#"
fn main() {
$0
}
            "#,
        );
    }

    #[test]
    fn adds_space_after_return_type() {
        type_char(
            '>',
            r#"
fn foo() -$0{ 92 }
"#,
            r#"
fn foo() -> { 92 }
"#,
        );
    }

    #[test]
    fn adds_closing_brace_for_expr() {
        type_char(
            '{',
            r#"
fn f() { match () { _ => $0() } }
            "#,
            r#"
fn f() { match () { _ => {()} } }
            "#,
        );
        type_char(
            '{',
            r#"
fn f() { $0() }
            "#,
            r#"
fn f() { {()} }
            "#,
        );
        type_char(
            '{',
            r#"
fn f() { let x = $0(); }
            "#,
            r#"
fn f() { let x = {()}; }
            "#,
        );
        type_char(
            '{',
            r#"
fn f() { let x = $0a.b(); }
            "#,
            r#"
fn f() { let x = {a.b()}; }
            "#,
        );
        type_char(
            '{',
            r#"
const S: () = $0();
fn f() {}
            "#,
            r#"
const S: () = {()};
fn f() {}
            "#,
        );
        type_char(
            '{',
            r#"
const S: () = $0a.b();
fn f() {}
            "#,
            r#"
const S: () = {a.b()};
fn f() {}
            "#,
        );
        type_char(
            '{',
            r#"
fn f() {
    match x {
        0 => $0(),
        1 => (),
    }
}
            "#,
            r#"
fn f() {
    match x {
        0 => {()},
        1 => (),
    }
}
            "#,
        );
    }

    #[test]
    fn noop_in_string_literal() {
        // Regression test for #9351
        type_char_noop(
            '{',
            r##"
fn check_with(ra_fixture: &str, expect: Expect) {
    let base = r#"
enum E { T(), R$0, C }
use self::E::X;
const Z: E = E::C;
mod m {}
asdasdasdasdasdasda
sdasdasdasdasdasda
sdasdasdasdasd
"#;
    let actual = completion_list(&format!("{}\n{}", base, ra_fixture));
    expect.assert_eq(&actual)
}
            "##,
        );
    }

    #[test]
    fn noop_in_item_position_with_macro() {
        type_char_noop('{', r#"$0println!();"#);
        type_char_noop(
            '{',
            r#"
fn main() $0println!("hello");
}"#,
        );
    }

    #[test]
    fn adds_closing_brace_for_use_tree() {
        type_char(
            '{',
            r#"
use some::$0Path;
            "#,
            r#"
use some::{Path};
            "#,
        );
        type_char(
            '{',
            r#"
use some::{Path, $0Other};
            "#,
            r#"
use some::{Path, {Other}};
            "#,
        );
        type_char(
            '{',
            r#"
use some::{$0Path, Other};
            "#,
            r#"
use some::{{Path}, Other};
            "#,
        );
        type_char(
            '{',
            r#"
use some::path::$0to::Item;
            "#,
            r#"
use some::path::{to::Item};
            "#,
        );
        type_char(
            '{',
            r#"
use some::$0path::to::Item;
            "#,
            r#"
use some::{path::to::Item};
            "#,
        );
        type_char(
            '{',
            r#"
use $0some::path::to::Item;
            "#,
            r#"
use {some::path::to::Item};
            "#,
        );
        type_char(
            '{',
            r#"
use some::path::$0to::{Item};
            "#,
            r#"
use some::path::{to::{Item}};
            "#,
        );
        type_char(
            '{',
            r#"
use $0Thing as _;
            "#,
            r#"
use {Thing as _};
            "#,
        );

        type_char_noop(
            '{',
            r#"
use some::pa$0th::to::Item;
            "#,
        );
    }

    #[test]
    fn adds_closing_angle_bracket_for_generic_args() {
        type_char(
            '<',
            r#"
fn foo() {
    bar::$0
}
            "#,
            r#"
fn foo() {
    bar::<$0>
}
            "#,
        );

        type_char(
            '<',
            r#"
fn foo(bar: &[u64]) {
    bar.iter().collect::$0();
}
            "#,
            r#"
fn foo(bar: &[u64]) {
    bar.iter().collect::<$0>();
}
            "#,
        );
    }

    #[test]
    fn adds_closing_angle_bracket_for_generic_params() {
        type_char(
            '<',
            r#"
fn foo$0() {}
            "#,
            r#"
fn foo<$0>() {}
            "#,
        );
        type_char(
            '<',
            r#"
fn foo$0
            "#,
            r#"
fn foo<$0>
            "#,
        );
        type_char(
            '<',
            r#"
struct Foo$0 {}
            "#,
            r#"
struct Foo<$0> {}
            "#,
        );
        type_char(
            '<',
            r#"
struct Foo$0();
            "#,
            r#"
struct Foo<$0>();
            "#,
        );
        type_char(
            '<',
            r#"
struct Foo$0
            "#,
            r#"
struct Foo<$0>
            "#,
        );
        type_char(
            '<',
            r#"
enum Foo$0
            "#,
            r#"
enum Foo<$0>
            "#,
        );
        type_char(
            '<',
            r#"
trait Foo$0
            "#,
            r#"
trait Foo<$0>
            "#,
        );
        type_char(
            '<',
            r#"
type Foo$0 = Bar;
            "#,
            r#"
type Foo<$0> = Bar;
            "#,
        );
        type_char(
            '<',
            r#"
impl$0 Foo {}
            "#,
            r#"
impl<$0> Foo {}
            "#,
        );
        type_char(
            '<',
            r#"
impl<T> Foo$0 {}
            "#,
            r#"
impl<T> Foo<$0> {}
            "#,
        );
        type_char(
            '<',
            r#"
impl Foo$0 {}
            "#,
            r#"
impl Foo<$0> {}
            "#,
        );
    }

    #[test]
    fn dont_add_closing_angle_bracket_for_comparison() {
        type_char_noop(
            '<',
            r#"
fn main() {
    42$0
}
            "#,
        );
        type_char_noop(
            '<',
            r#"
fn main() {
    42 $0
}
            "#,
        );
        type_char_noop(
            '<',
            r#"
fn main() {
    let foo = 42;
    foo $0
}
            "#,
        );
    }

    #[test]
    fn dont_add_closing_angle_bracket_if_it_is_already_there() {
        type_char_noop(
            '<',
            r#"
fn foo() {
    bar::$0>
}
            "#,
        );
        type_char_noop(
            '<',
            r#"
fn foo(bar: &[u64]) {
    bar.iter().collect::$0   >();
}
            "#,
        );
        type_char_noop(
            '<',
            r#"
fn foo$0>() {}
            "#,
        );
        type_char_noop(
            '<',
            r#"
fn foo$0>
            "#,
        );
        type_char_noop(
            '<',
            r#"
struct Foo$0> {}
            "#,
        );
        type_char_noop(
            '<',
            r#"
struct Foo$0>();
            "#,
        );
        type_char_noop(
            '<',
            r#"
struct Foo$0>
            "#,
        );
        type_char_noop(
            '<',
            r#"
enum Foo$0>
            "#,
        );
        type_char_noop(
            '<',
            r#"
trait Foo$0>
            "#,
        );
        type_char_noop(
            '<',
            r#"
type Foo$0> = Bar;
            "#,
        );
        type_char_noop(
            '<',
            r#"
impl$0> Foo {}
            "#,
        );
        type_char_noop(
            '<',
            r#"
impl<T> Foo$0> {}
            "#,
        );
        type_char_noop(
            '<',
            r#"
impl Foo$0> {}
            "#,
        );
    }

    #[test]
    fn regression_629() {
        type_char_noop(
            '.',
            r#"
fn foo() {
    CompletionItem::new(
        CompletionKind::Reference,
        ctx.source_range(),
        field.name().to_string(),
    )
    .foo()
    $0
}
"#,
        );
        type_char_noop(
            '.',
            r#"
fn foo() {
    CompletionItem::new(
        CompletionKind::Reference,
        ctx.source_range(),
        field.name().to_string(),
    )
    $0
}
"#,
        );
    }
}
