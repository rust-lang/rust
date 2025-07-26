use hir::db::ExpandDatabase;
use ide_db::{assists::Assist, base_db::AnchoredPathBuf, source_change::FileSystemEdit};
use itertools::Itertools;
use syntax::AstNode;

use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext, fix};

// Diagnostic: unresolved-module
//
// This diagnostic is triggered if rust-analyzer is unable to discover referred module.
pub(crate) fn unresolved_module(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::UnresolvedModule,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0583"),
        match &*d.candidates {
            [] => "unresolved module".to_owned(),
            [candidate] => format!("unresolved module, can't find module file: {candidate}"),
            [candidates @ .., last] => {
                format!(
                    "unresolved module, can't find module file: {}, or {}",
                    candidates.iter().format(", "),
                    last
                )
            }
        },
        d.decl.map(|it| it.into()),
    )
    .stable()
    .with_fixes(fixes(ctx, d))
}

fn fixes(ctx: &DiagnosticsContext<'_>, d: &hir::UnresolvedModule) -> Option<Vec<Assist>> {
    let root = ctx.sema.db.parse_or_expand(d.decl.file_id);
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
                            anchor: d.decl.file_id.original_file(ctx.sema.db).file_id(ctx.sema.db),
                            path: candidate.clone(),
                        },
                        initial_contents: "".to_owned(),
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
    use crate::tests::check_diagnostics;

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
        check_diagnostics(
            r#"
  mod foo;
//^^^^^^^^ 💡 error: unresolved module, can't find module file: foo.rs, or foo/mod.rs
"#,
        );
    }
}
