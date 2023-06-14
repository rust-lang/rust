//! Suggests shortening `Foo { field: field }` to `Foo { field }` in both
//! expressions and patterns.

use ide_db::{base_db::FileId, source_change::SourceChange};
use syntax::{ast, match_ast, AstNode, SyntaxNode};
use text_edit::TextEdit;

use crate::{fix, Diagnostic, DiagnosticCode};

pub(crate) fn field_shorthand(acc: &mut Vec<Diagnostic>, file_id: FileId, node: &SyntaxNode) {
    match_ast! {
        match node {
            ast::RecordExpr(it) => check_expr_field_shorthand(acc, file_id, it),
            ast::RecordPat(it) => check_pat_field_shorthand(acc, file_id, it),
            _ => ()
        }
    };
}

fn check_expr_field_shorthand(
    acc: &mut Vec<Diagnostic>,
    file_id: FileId,
    record_expr: ast::RecordExpr,
) {
    let record_field_list = match record_expr.record_expr_field_list() {
        Some(it) => it,
        None => return,
    };
    for record_field in record_field_list.fields() {
        let (name_ref, expr) = match record_field.name_ref().zip(record_field.expr()) {
            Some(it) => it,
            None => continue,
        };

        let field_name = name_ref.syntax().text().to_string();
        let field_expr = expr.syntax().text().to_string();
        let field_name_is_tup_index = name_ref.as_tuple_field().is_some();
        if field_name != field_expr || field_name_is_tup_index {
            continue;
        }

        let mut edit_builder = TextEdit::builder();
        edit_builder.delete(record_field.syntax().text_range());
        edit_builder.insert(record_field.syntax().text_range().start(), field_name);
        let edit = edit_builder.finish();

        let field_range = record_field.syntax().text_range();
        acc.push(
            Diagnostic::new(
                DiagnosticCode::Clippy("redundant_field_names"),
                "Shorthand struct initialization",
                field_range,
            )
            .with_fixes(Some(vec![fix(
                "use_expr_field_shorthand",
                "Use struct shorthand initialization",
                SourceChange::from_text_edit(file_id, edit),
                field_range,
            )])),
        );
    }
}

fn check_pat_field_shorthand(
    acc: &mut Vec<Diagnostic>,
    file_id: FileId,
    record_pat: ast::RecordPat,
) {
    let record_pat_field_list = match record_pat.record_pat_field_list() {
        Some(it) => it,
        None => return,
    };
    for record_pat_field in record_pat_field_list.fields() {
        let (name_ref, pat) = match record_pat_field.name_ref().zip(record_pat_field.pat()) {
            Some(it) => it,
            None => continue,
        };

        let field_name = name_ref.syntax().text().to_string();
        let field_pat = pat.syntax().text().to_string();
        let field_name_is_tup_index = name_ref.as_tuple_field().is_some();
        if field_name != field_pat || field_name_is_tup_index {
            continue;
        }

        let mut edit_builder = TextEdit::builder();
        edit_builder.delete(record_pat_field.syntax().text_range());
        edit_builder.insert(record_pat_field.syntax().text_range().start(), field_name);
        let edit = edit_builder.finish();

        let field_range = record_pat_field.syntax().text_range();
        acc.push(
            Diagnostic::new(
                DiagnosticCode::Clippy("redundant_field_names"),
                "Shorthand struct pattern",
                field_range,
            )
            .with_fixes(Some(vec![fix(
                "use_pat_field_shorthand",
                "Use struct field shorthand",
                SourceChange::from_text_edit(file_id, edit),
                field_range,
            )])),
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_diagnostics, check_fix};

    #[test]
    fn test_check_expr_field_shorthand() {
        check_diagnostics(
            r#"
struct A { a: &'static str }
fn main() { A { a: "hello" }; }
"#,
        );
        check_diagnostics(
            r#"
struct A(usize);
fn main() { A { 0: 0 }; }
"#,
        );

        check_fix(
            r#"
struct A { a: &'static str }
fn main() {
    let a = "haha";
    A { a$0: a };
}
"#,
            r#"
struct A { a: &'static str }
fn main() {
    let a = "haha";
    A { a };
}
"#,
        );

        check_fix(
            r#"
struct A { a: &'static str, b: &'static str }
fn main() {
    let a = "haha";
    let b = "bb";
    A { a$0: a, b };
}
"#,
            r#"
struct A { a: &'static str, b: &'static str }
fn main() {
    let a = "haha";
    let b = "bb";
    A { a, b };
}
"#,
        );
    }

    #[test]
    fn test_check_pat_field_shorthand() {
        check_diagnostics(
            r#"
struct A { a: &'static str }
fn f(a: A) { let A { a: hello } = a; }
"#,
        );
        check_diagnostics(
            r#"
struct A(usize);
fn f(a: A) { let A { 0: 0 } = a; }
"#,
        );

        check_fix(
            r#"
struct A { a: &'static str }
fn f(a: A) {
    let A { a$0: a } = a;
}
"#,
            r#"
struct A { a: &'static str }
fn f(a: A) {
    let A { a } = a;
}
"#,
        );

        check_fix(
            r#"
struct A { a: &'static str, b: &'static str }
fn f(a: A) {
    let A { a$0: a, b } = a;
}
"#,
            r#"
struct A { a: &'static str, b: &'static str }
fn f(a: A) {
    let A { a, b } = a;
}
"#,
        );
    }
}
