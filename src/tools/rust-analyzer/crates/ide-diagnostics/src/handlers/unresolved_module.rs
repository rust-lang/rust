use hir::db::ExpandDatabase;
use ide_db::{assists::Assist, base_db::AnchoredPathBuf, source_change::FileSystemEdit};
use itertools::Itertools;
use syntax::AstNode;

use crate::{fix, Diagnostic, DiagnosticsContext};

// Diagnostic: unresolved-module
//
// This diagnostic is triggered if rust-analyzer is unable to discover referred module.
pub(crate) fn unresolved_module(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::UnresolvedModule,
) -> Diagnostic {
    Diagnostic::new(
        "unresolved-module",
        match &*d.candidates {
            [] => "unresolved module".to_string(),
            [candidate] => format!("unresolved module, can't find module file: {candidate}"),
            [candidates @ .., last] => {
                format!(
                    "unresolved module, can't find module file: {}, or {}",
                    candidates.iter().format(", "),
                    last
                )
            }
        },
        ctx.sema.diagnostics_display_range(d.decl.clone().map(|it| it.into())).range,
    )
    .with_fixes(fixes(ctx, d))
}

fn fixes(ctx: &DiagnosticsContext<'_>, d: &hir::UnresolvedModule) -> Option<Vec<Assist>> {
    let root = ctx.sema.db.parse_or_expand(d.decl.file_id)?;
    let unresolved_module = d.decl.value.to_node(&root);
    Some(
        d.candidates
            .iter()
            .map(|candidate| {
                fix(
                    "create_module",
                    &format!("Create module at `{candidate}`"),
                    FileSystemEdit::CreateFile {
                        dst: AnchoredPathBuf {
                            anchor: d.decl.file_id.original_file(ctx.sema.db),
                            path: candidate.clone(),
                        },
                        initial_contents: "".to_string(),
                    }
                    .into(),
                    unresolved_module.syntax().text_range(),
                )
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::tests::{check_diagnostics, check_expect};

    #[test]
    fn unresolved_module() {
        check_diagnostics(
            r#"
//- /lib.rs
mod foo;
  mod bar;
//^^^^^^^^ 💡 error: unresolved module, can't find module file: bar.rs, or bar/mod.rs
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
                        code: DiagnosticCode(
                            "unresolved-module",
                        ),
                        message: "unresolved module, can't find module file: foo.rs, or foo/mod.rs",
                        range: 0..8,
                        severity: Error,
                        unused: false,
                        experimental: false,
                        fixes: Some(
                            [
                                Assist {
                                    id: AssistId(
                                        "create_module",
                                        QuickFix,
                                    ),
                                    label: "Create module at `foo.rs`",
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
                                    trigger_signature_help: false,
                                },
                                Assist {
                                    id: AssistId(
                                        "create_module",
                                        QuickFix,
                                    ),
                                    label: "Create module at `foo/mod.rs`",
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
                                                        path: "foo/mod.rs",
                                                    },
                                                    initial_contents: "",
                                                },
                                            ],
                                            is_snippet: false,
                                        },
                                    ),
                                    trigger_signature_help: false,
                                },
                            ],
                        ),
                    },
                ]
            "#]],
        );
    }
}
