use hir::db::AstDatabase;
use ide_assists::Assist;
use ide_db::{base_db::AnchoredPathBuf, source_change::FileSystemEdit};
use syntax::AstNode;

use crate::diagnostics::{fix, Diagnostic, DiagnosticsContext};

// Diagnostic: unresolved-module
//
// This diagnostic is triggered if rust-analyzer is unable to discover referred module.
pub(super) fn unresolved_module(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::UnresolvedModule,
) -> Diagnostic {
    Diagnostic::new(
        "unresolved-module",
        "unresolved module",
        ctx.sema.diagnostics_display_range(d.decl.clone().map(|it| it.into())).range,
    )
    .with_fixes(fixes(ctx, d))
}

fn fixes(ctx: &DiagnosticsContext<'_>, d: &hir::UnresolvedModule) -> Option<Vec<Assist>> {
    let root = ctx.sema.db.parse_or_expand(d.decl.file_id)?;
    let unresolved_module = d.decl.value.to_node(&root);
    Some(vec![fix(
        "create_module",
        "Create module",
        FileSystemEdit::CreateFile {
            dst: AnchoredPathBuf {
                anchor: d.decl.file_id.original_file(ctx.sema.db),
                path: d.candidate.clone(),
            },
            initial_contents: "".to_string(),
        }
        .into(),
        unresolved_module.syntax().text_range(),
    )])
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::diagnostics::tests::{check_diagnostics, check_expect};

    #[test]
    fn unresolved_module() {
        check_diagnostics(
            r#"
//- /lib.rs
mod foo;
  mod bar;
//^^^^^^^^ unresolved module
mod baz {}
//- /foo.rs
"#,
        );
    }

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
