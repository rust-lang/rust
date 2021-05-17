use hir::{db::AstDatabase, diagnostics::ReplaceFilterMapNextWithFindMap, Semantics};
use ide_assists::{Assist, AssistResolveStrategy};
use ide_db::{source_change::SourceChange, RootDatabase};
use syntax::{
    ast::{self, ArgListOwner},
    AstNode, TextRange,
};
use text_edit::TextEdit;

use crate::diagnostics::{fix, DiagnosticWithFix};

impl DiagnosticWithFix for ReplaceFilterMapNextWithFindMap {
    fn fix(
        &self,
        sema: &Semantics<RootDatabase>,
        _resolve: &AssistResolveStrategy,
    ) -> Option<Assist> {
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

        Some(fix(
            "replace_with_find_map",
            "Replace filter_map(..).next() with find_map()",
            source_change,
            trigger_range,
        ))
    }
}
