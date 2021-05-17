use hir::{db::AstDatabase, diagnostics::RemoveThisSemicolon, Semantics};
use ide_assists::{Assist, AssistResolveStrategy};
use ide_db::{source_change::SourceChange, RootDatabase};
use syntax::{ast, AstNode};
use text_edit::TextEdit;

use crate::diagnostics::{fix, DiagnosticWithFix};

impl DiagnosticWithFix for RemoveThisSemicolon {
    fn fix(
        &self,
        sema: &Semantics<RootDatabase>,
        _resolve: &AssistResolveStrategy,
    ) -> Option<Assist> {
        let root = sema.db.parse_or_expand(self.file)?;

        let semicolon = self
            .expr
            .to_node(&root)
            .syntax()
            .parent()
            .and_then(ast::ExprStmt::cast)
            .and_then(|expr| expr.semicolon_token())?
            .text_range();

        let edit = TextEdit::delete(semicolon);
        let source_change = SourceChange::from_text_edit(self.file.original_file(sema.db), edit);

        Some(fix("remove_semicolon", "Remove this semicolon", source_change, semicolon))
    }
}
