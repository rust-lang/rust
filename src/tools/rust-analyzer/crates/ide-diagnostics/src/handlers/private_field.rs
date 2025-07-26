use hir::{EditionedFileId, FileRange, HasCrate, HasSource, Semantics};
use ide_db::{RootDatabase, assists::Assist, source_change::SourceChange, text_edit::TextEdit};
use syntax::{AstNode, TextRange, TextSize, ast::HasVisibility};

use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext, fix};

// Diagnostic: private-field
//
// This diagnostic is triggered if the accessed field is not visible from the current module.
pub(crate) fn private_field(ctx: &DiagnosticsContext<'_>, d: &hir::PrivateField) -> Diagnostic {
    // FIXME: add quickfix
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0616"),
        format!(
            "field `{}` of `{}` is private",
            d.field.name(ctx.sema.db).display(ctx.sema.db, ctx.edition),
            d.field.parent_def(ctx.sema.db).name(ctx.sema.db).display(ctx.sema.db, ctx.edition)
        ),
        d.expr.map(|it| it.into()),
    )
    .stable()
    .with_fixes(field_is_private_fixes(
        &ctx.sema,
        d.expr.file_id.original_file(ctx.sema.db),
        d.field,
        ctx.sema.original_range(d.expr.to_node(ctx.sema.db).syntax()).range,
    ))
}

pub(crate) fn field_is_private_fixes(
    sema: &Semantics<'_, RootDatabase>,
    usage_file_id: EditionedFileId,
    private_field: hir::Field,
    fix_range: TextRange,
) -> Option<Vec<Assist>> {
    let def_crate = private_field.krate(sema.db);
    let usage_crate = sema.file_to_module_def(usage_file_id.file_id(sema.db))?.krate();
    let mut visibility_text = if usage_crate == def_crate { "pub(crate) " } else { "pub " };

    let source = private_field.source(sema.db)?;
    let existing_visibility = match &source.value {
        hir::FieldSource::Named(it) => it.visibility(),
        hir::FieldSource::Pos(it) => it.visibility(),
    };
    let range = match existing_visibility {
        Some(visibility) => {
            // If there is an existing visibility, don't insert whitespace after.
            visibility_text = visibility_text.trim_end();
            source.with_value(visibility.syntax()).original_file_range_opt(sema.db)?.0
        }
        None => {
            let (range, _) = source.syntax().original_file_range_opt(sema.db)?;
            FileRange {
                file_id: range.file_id,
                range: TextRange::at(range.range.start(), TextSize::new(0)),
            }
        }
    };
    let source_change = SourceChange::from_text_edit(
        range.file_id.file_id(sema.db),
        TextEdit::replace(range.range, visibility_text.into()),
    );

    Some(vec![fix(
        "increase_field_visibility",
        "Increase field visibility",
        source_change,
        fix_range,
    )])
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_diagnostics, check_fix};

    #[test]
    fn private_field() {
        check_diagnostics(
            r#"
mod module { pub struct Struct { field: u32 } }
fn main(s: module::Struct) {
    s.field;
  //^^^^^^^ 💡 error: field `field` of `Struct` is private
}
"#,
        );
    }

    #[test]
    fn private_tuple_field() {
        check_diagnostics(
            r#"
mod module { pub struct Struct(u32); }
fn main(s: module::Struct) {
    s.0;
  //^^^ 💡 error: field `0` of `Struct` is private
}
"#,
        );
    }

    #[test]
    fn private_but_shadowed_in_deref() {
        check_diagnostics(
            r#"
//- minicore: deref
mod module {
    pub struct Struct { field: Inner }
    pub struct Inner { pub field: u32 }
    impl core::ops::Deref for Struct {
        type Target = Inner;
        fn deref(&self) -> &Inner { &self.field }
    }
}
fn main(s: module::Struct) {
    s.field;
}
"#,
        );
    }

    #[test]
    fn block_module_madness() {
        check_diagnostics(
            r#"
fn main() {
    let strukt = {
        use crate as ForceParentBlockDefMap;
        {
            pub struct Struct {
                field: (),
            }
            Struct { field: () }
        }
    };
    strukt.field;
}
"#,
        );
    }

    #[test]
    fn block_module_madness2() {
        check_diagnostics(
            r#"
fn main() {
    use crate as ForceParentBlockDefMap;
    let strukt = {
        use crate as ForceParentBlockDefMap;
        {
            pub struct Struct {
                field: (),
            }
            {
                use crate as ForceParentBlockDefMap;
                {
                    Struct { field: () }
                }
            }
        }
    };
    strukt.field;
}
"#,
        );
    }

    #[test]
    fn change_visibility_fix() {
        check_fix(
            r#"
pub mod foo {
    pub mod bar {
        pub struct Struct {
            field: i32,
        }
    }
}

fn foo(v: foo::bar::Struct) {
    v.field$0;
}
            "#,
            r#"
pub mod foo {
    pub mod bar {
        pub struct Struct {
            pub(crate) field: i32,
        }
    }
}

fn foo(v: foo::bar::Struct) {
    v.field;
}
            "#,
        );
    }

    #[test]
    fn change_visibility_with_existing_visibility() {
        check_fix(
            r#"
pub mod foo {
    pub mod bar {
        pub struct Struct {
            pub(super) field: i32,
        }
    }
}

fn foo(v: foo::bar::Struct) {
    v.field$0;
}
            "#,
            r#"
pub mod foo {
    pub mod bar {
        pub struct Struct {
            pub(crate) field: i32,
        }
    }
}

fn foo(v: foo::bar::Struct) {
    v.field;
}
            "#,
        );
    }
}
