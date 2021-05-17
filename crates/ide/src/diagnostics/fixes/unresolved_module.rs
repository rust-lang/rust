use hir::{db::AstDatabase, diagnostics::UnresolvedModule, Semantics};
use ide_assists::{Assist, AssistResolveStrategy};
use ide_db::{base_db::AnchoredPathBuf, source_change::FileSystemEdit, RootDatabase};
use syntax::AstNode;

use crate::diagnostics::{fix, DiagnosticWithFixes};

impl DiagnosticWithFixes for UnresolvedModule {
    fn fixes(
        &self,
        sema: &Semantics<RootDatabase>,
        _resolve: &AssistResolveStrategy,
    ) -> Option<Vec<Assist>> {
        let root = sema.db.parse_or_expand(self.file)?;
        let unresolved_module = self.decl.to_node(&root);
        Some(vec![fix(
            "create_module",
            "Create module",
            FileSystemEdit::CreateFile {
                dst: AnchoredPathBuf {
                    anchor: self.file.original_file(sema.db),
                    path: self.candidate.clone(),
                },
                initial_contents: "".to_string(),
            }
            .into(),
            unresolved_module.syntax().text_range(),
        )])
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::diagnostics::tests::check_expect;

    #[test]
    fn test_unresolved_module_diagnostic() {
        check_expect(
            r#"mod foo;"#,
            expect![[r#"
                [
                    Diagnostic {
                        message: "unresolved module",
                        range: 0..8,
                        severity: Error,
                        fixes: Some(
                            [
                                Assist {
                                    id: AssistId(
                                        "create_module",
                                        QuickFix,
                                    ),
                                    label: "Create module",
                                    group: None,
                                    target: 0..8,
                                    source_change: Some(
                                        SourceChange {
                                            source_file_edits: {},
                                            file_system_edits: [
                                                CreateFile {
                                                    dst: AnchoredPathBuf {
                                                        anchor: FileId(
                                                            0,
                                                        ),
                                                        path: "foo.rs",
                                                    },
                                                    initial_contents: "",
                                                },
                                            ],
                                            is_snippet: false,
                                        },
                                    ),
                                },
                            ],
                        ),
                        unused: false,
                        code: Some(
                            DiagnosticCode(
                                "unresolved-module",
                            ),
                        ),
                    },
                ]
            "#]],
        );
    }
}
