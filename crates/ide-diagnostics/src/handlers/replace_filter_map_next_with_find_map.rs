use hir::{db::AstDatabase, InFile};
use ide_db::source_change::SourceChange;
use syntax::{
    ast::{self, HasArgList},
    AstNode, TextRange,
};
use text_edit::TextEdit;

use crate::{fix, Assist, Diagnostic, DiagnosticsContext, Severity};

// Diagnostic: replace-filter-map-next-with-find-map
//
// This diagnostic is triggered when `.filter_map(..).next()` is used, rather than the more concise `.find_map(..)`.
pub(crate) fn replace_filter_map_next_with_find_map(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::ReplaceFilterMapNextWithFindMap,
) -> Diagnostic {
    Diagnostic::new(
        "replace-filter-map-next-with-find-map",
        "replace filter_map(..).next() with find_map(..)",
        ctx.sema.diagnostics_display_range(InFile::new(d.file, d.next_expr.clone().into())).range,
    )
    .severity(Severity::WeakWarning)
    .with_fixes(fixes(ctx, d))
}

fn fixes(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::ReplaceFilterMapNextWithFindMap,
) -> Option<Vec<Assist>> {
    let root = ctx.sema.db.parse_or_expand(d.file)?;
    let next_expr = d.next_expr.to_node(&root);
    let next_call = ast::MethodCallExpr::cast(next_expr.syntax().clone())?;

    let filter_map_call = ast::MethodCallExpr::cast(next_call.receiver()?.syntax().clone())?;
    let filter_map_name_range = filter_map_call.name_ref()?.ident_token()?.text_range();
    let filter_map_args = filter_map_call.arg_list()?;

    let range_to_replace =
        TextRange::new(filter_map_name_range.start(), next_expr.syntax().text_range().end());
    let replacement = format!("find_map{}", filter_map_args.syntax().text());
    let trigger_range = next_expr.syntax().text_range();

    let edit = TextEdit::replace(range_to_replace, replacement);

    let source_change = SourceChange::from_text_edit(d.file.original_file(ctx.sema.db), edit);

    Some(vec![fix(
        "replace_with_find_map",
        "Replace filter_map(..).next() with find_map()",
        source_change,
        trigger_range,
    )])
}

#[cfg(test)]
mod tests {
    use crate::{
        tests::{check_diagnostics_with_config, check_fix},
        DiagnosticsConfig,
    };

    #[track_caller]
    pub(crate) fn check_diagnostics(ra_fixture: &str) {
        let mut config = DiagnosticsConfig::test_sample();
        config.disabled.insert("inactive-code".to_string());
        config.disabled.insert("unresolved-method".to_string());
        check_diagnostics_with_config(config, ra_fixture)
    }

    #[test]
    fn replace_filter_map_next_with_find_map2() {
        check_diagnostics(
            r#"
//- minicore: iterators
fn foo() {
    let m = core::iter::repeat(()).filter_map(|()| Some(92)).next();
}         //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 💡 weak: replace filter_map(..).next() with find_map(..)
"#,
        );
    }

    #[test]
    fn replace_filter_map_next_with_find_map_no_diagnostic_without_next() {
        check_diagnostics(
            r#"
//- minicore: iterators
fn foo() {
    let m = core::iter::repeat(())
        .filter_map(|()| Some(92))
        .count();
}
"#,
        );
    }

    #[test]
    fn replace_filter_map_next_with_find_map_no_diagnostic_with_intervening_methods() {
        check_diagnostics(
            r#"
//- minicore: iterators
fn foo() {
    let m = core::iter::repeat(())
        .filter_map(|()| Some(92))
        .map(|x| x + 2)
        .next();
}
"#,
        );
    }

    #[test]
    fn replace_filter_map_next_with_find_map_no_diagnostic_if_not_in_chain() {
        check_diagnostics(
            r#"
//- minicore: iterators
fn foo() {
    let m = core::iter::repeat(())
        .filter_map(|()| Some(92));
    let n = m.next();
}
"#,
        );
    }

    #[test]
    fn replace_with_find_map() {
        check_fix(
            r#"
//- minicore: iterators
fn foo() {
    let m = core::iter::repeat(()).$0filter_map(|()| Some(92)).next();
}
"#,
            r#"
fn foo() {
    let m = core::iter::repeat(()).find_map(|()| Some(92));
}
"#,
        )
    }
}
