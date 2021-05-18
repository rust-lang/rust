use hir::{db::AstDatabase, diagnostics::ReplaceFilterMapNextWithFindMap, Semantics};
use ide_assists::{Assist, AssistResolveStrategy};
use ide_db::{source_change::SourceChange, RootDatabase};
use syntax::{
    ast::{self, ArgListOwner},
    AstNode, TextRange,
};
use text_edit::TextEdit;

use crate::diagnostics::{fix, DiagnosticWithFixes};

impl DiagnosticWithFixes for ReplaceFilterMapNextWithFindMap {
    fn fixes(
        &self,
        sema: &Semantics<RootDatabase>,
        _resolve: &AssistResolveStrategy,
    ) -> Option<Vec<Assist>> {
        let root = sema.db.parse_or_expand(self.file)?;
        let next_expr = self.next_expr.to_node(&root);
        let next_call = ast::MethodCallExpr::cast(next_expr.syntax().clone())?;

        let filter_map_call = ast::MethodCallExpr::cast(next_call.receiver()?.syntax().clone())?;
        let filter_map_name_range = filter_map_call.name_ref()?.ident_token()?.text_range();
        let filter_map_args = filter_map_call.arg_list()?;

        let range_to_replace =
            TextRange::new(filter_map_name_range.start(), next_expr.syntax().text_range().end());
        let replacement = format!("find_map{}", filter_map_args.syntax().text());
        let trigger_range = next_expr.syntax().text_range();

        let edit = TextEdit::replace(range_to_replace, replacement);

        let source_change = SourceChange::from_text_edit(self.file.original_file(sema.db), edit);

        Some(vec![fix(
            "replace_with_find_map",
            "Replace filter_map(..).next() with find_map()",
            source_change,
            trigger_range,
        )])
    }
}

#[cfg(test)]
mod tests {
    use crate::diagnostics::tests::check_fix;

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
