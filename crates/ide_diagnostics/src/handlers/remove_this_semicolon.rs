use ide_db::{
    base_db::{FileLoader, FileRange},
    source_change::SourceChange,
};
use syntax::{TextRange, TextSize};
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
        semicolon_range(ctx, d).unwrap_or_else(|it| it).range,
    )
    .with_fixes(fixes(ctx, d))
}

fn semicolon_range(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::RemoveThisSemicolon,
) -> Result<FileRange, FileRange> {
    let expr_range = ctx.sema.diagnostics_display_range(d.expr.clone().map(|it| it.into()));
    let file_text = ctx.sema.db.file_text(expr_range.file_id);
    let range_end: usize = expr_range.range.end().into();
    // FIXME: This doesn't handle whitespace and comments, but handling those in
    // the presence of macros might prove tricky...
    if file_text[range_end..].starts_with(';') {
        Ok(FileRange {
            file_id: expr_range.file_id,
            range: TextRange::at(expr_range.range.end(), TextSize::of(';')),
        })
    } else {
        Err(expr_range)
    }
}

fn fixes(ctx: &DiagnosticsContext<'_>, d: &hir::RemoveThisSemicolon) -> Option<Vec<Assist>> {
    let semicolon_range = semicolon_range(ctx, d).ok()?;

    let edit = TextEdit::delete(semicolon_range.range);
    let source_change = SourceChange::from_text_edit(semicolon_range.file_id, edit);

    Some(vec![fix(
        "remove_semicolon",
        "Remove this semicolon",
        source_change,
        semicolon_range.range,
    )])
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_diagnostics, check_fix};

    #[test]
    fn missing_semicolon() {
        check_diagnostics(
            r#"
fn test() -> i32 { 123; }
                    //^ 💡 error: remove this semicolon
"#,
        );
    }

    #[test]
    fn remove_semicolon() {
        check_fix(r#"fn f() -> i32 { 92$0; }"#, r#"fn f() -> i32 { 92 }"#);
    }
}
