use hir::{db::AstDatabase, InFile};
use ide_db::source_change::SourceChange;
use syntax::{
    ast::{self, ArgListOwner},
    AstNode, TextRange,
};
use text_edit::TextEdit;

use crate::{fix, Assist, Diagnostic, DiagnosticsContext, Severity};

// Diagnostic: replace-filter-map-next-with-find-map
//
// This diagnostic is triggered when `.filter_map(..).next()` is used, rather than the more concise `.find_map(..)`.
pub(super) fn replace_filter_map_next_with_find_map(
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
    use crate::tests::check_fix;

    // Register the required standard library types to make the tests work
    #[track_caller]
    fn check_diagnostics(ra_fixture: &str) {
        let prefix = r#"
//- /main.rs crate:main deps:core
use core::iter::Iterator;
use core::option::Option::{self, Some, None};
"#;
        let suffix = r#"
//- /core/lib.rs crate:core
pub mod option {
    pub enum Option<T> { Some(T), None }
}
pub mod iter {
    pub trait Iterator {
        type Item;
        fn filter_map<B, F>(self, f: F) -> FilterMap where F: FnMut(Self::Item) -> Option<B> { FilterMap }
        fn next(&mut self) -> Option<Self::Item>;
    }
    pub struct FilterMap {}
    impl Iterator for FilterMap {
        type Item = i32;
        fn next(&mut self) -> i32 { 7 }
    }
}
"#;
        crate::tests::check_diagnostics(&format!("{}{}{}", prefix, ra_fixture, suffix))
    }

    #[test]
    fn replace_filter_map_next_with_find_map2() {
        check_diagnostics(
            r#"
    fn foo() {
        let m = [1, 2, 3].iter().filter_map(|x| if *x == 2 { Some (4) } else { None }).next();
    }         //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ replace filter_map(..).next() with find_map(..)
"#,
        );
    }

    #[test]
    fn replace_filter_map_next_with_find_map_no_diagnostic_without_next() {
        check_diagnostics(
            r#"
fn foo() {
    let m = [1, 2, 3]
        .iter()
        .filter_map(|x| if *x == 2 { Some (4) } else { None })
        .len();
}
"#,
        );
    }

    #[test]
    fn replace_filter_map_next_with_find_map_no_diagnostic_with_intervening_methods() {
        check_diagnostics(
            r#"
fn foo() {
    let m = [1, 2, 3]
        .iter()
        .filter_map(|x| if *x == 2 { Some (4) } else { None })
        .map(|x| x + 2)
        .len();
}
"#,
        );
    }

    #[test]
    fn replace_filter_map_next_with_find_map_no_diagnostic_if_not_in_chain() {
        check_diagnostics(
            r#"
fn foo() {
    let m = [1, 2, 3]
        .iter()
        .filter_map(|x| if *x == 2 { Some (4) } else { None });
    let n = m.next();
}
"#,
        );
    }

    #[test]
    fn replace_with_wind_map() {
        check_fix(
            r#"
//- /main.rs crate:main deps:core
use core::iter::Iterator;
use core::option::Option::{self, Some, None};
fn foo() {
    let m = [1, 2, 3].iter().$0filter_map(|x| if *x == 2 { Some (4) } else { None }).next();
}
//- /core/lib.rs crate:core
pub mod option {
    pub enum Option<T> { Some(T), None }
}
pub mod iter {
    pub trait Iterator {
        type Item;
        fn filter_map<B, F>(self, f: F) -> FilterMap where F: FnMut(Self::Item) -> Option<B> { FilterMap }
        fn next(&mut self) -> Option<Self::Item>;
    }
    pub struct FilterMap {}
    impl Iterator for FilterMap {
        type Item = i32;
        fn next(&mut self) -> i32 { 7 }
    }
}
"#,
            r#"
use core::iter::Iterator;
use core::option::Option::{self, Some, None};
fn foo() {
    let m = [1, 2, 3].iter().find_map(|x| if *x == 2 { Some (4) } else { None });
}
"#,
        )
    }
}
