use hir::db::AstDatabase;
use ide_db::source_change::SourceChange;
use syntax::{ast, AstNode};
use text_edit::TextEdit;

use crate::{fix, Assist, Diagnostic, DiagnosticsContext};

// Diagnostic: remove-this-semicolon
//
// This diagnostic is triggered when there's an erroneous `;` at the end of the block.
pub(crate) fn remove_this_semicolon(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::RemoveThisSemicolon,
) -> Diagnostic {
    Diagnostic::new(
        "remove-this-semicolon",
        "remove this semicolon",
        ctx.sema.diagnostics_display_range(d.expr.clone().map(|it| it.into())).range,
    )
    .with_fixes(fixes(ctx, d))
}

fn fixes(ctx: &DiagnosticsContext<'_>, d: &hir::RemoveThisSemicolon) -> Option<Vec<Assist>> {
    let root = ctx.sema.db.parse_or_expand(d.expr.file_id)?;

    let semicolon = d
        .expr
        .value
        .to_node(&root)
        .syntax()
        .parent()
        .and_then(ast::ExprStmt::cast)
        .and_then(|expr| expr.semicolon_token())?
        .text_range();

    let edit = TextEdit::delete(semicolon);
    let source_change =
        SourceChange::from_text_edit(d.expr.file_id.original_file(ctx.sema.db), edit);

    Some(vec![fix("remove_semicolon", "Remove this semicolon", source_change, semicolon)])
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_diagnostics, check_fix};

    #[test]
    fn missing_semicolon() {
        check_diagnostics(
            r#"
fn test() -> i32 { 123; }
                 //^^^ 💡 error: remove this semicolon
"#,
        );
    }

    #[test]
    fn remove_semicolon() {
        check_fix(r#"fn f() -> i32 { 92$0; }"#, r#"fn f() -> i32 { 92 }"#);
    }
}
