use hir::{db::ExpandDatabase, Adt, HasSource, HirDisplay, InFile};
use ide_db::{
    assists::{Assist, AssistId, AssistKind},
    base_db::FileRange,
    helpers::is_editable_crate,
    label::Label,
    source_change::{SourceChange, SourceChangeBuilder},
};
use syntax::{
    ast::{self, edit::IndentLevel, make},
    AstNode, AstPtr, SyntaxKind,
};
use text_edit::TextEdit;

use crate::{adjusted_display_range, Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: unresolved-field
//
// This diagnostic is triggered if a field does not exist on a given type.
pub(crate) fn unresolved_field(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::UnresolvedField,
) -> Diagnostic {
    let method_suffix = if d.method_with_same_name_exists {
        ", but a method with a similar name exists"
    } else {
        ""
    };
    Diagnostic::new(
        DiagnosticCode::RustcHardError("E0559"),
        format!(
            "no field `{}` on type `{}`{method_suffix}",
            d.name.display(ctx.sema.db),
            d.receiver.display(ctx.sema.db)
        ),
        adjusted_display_range(ctx, d.expr, &|expr| {
            Some(
                match expr {
                    ast::Expr::MethodCallExpr(it) => it.name_ref(),
                    ast::Expr::FieldExpr(it) => it.name_ref(),
                    _ => None,
                }?
                .syntax()
                .text_range(),
            )
        }),
    )
    .with_fixes(fixes(ctx, d))
    .experimental()
}

fn fixes(ctx: &DiagnosticsContext<'_>, d: &hir::UnresolvedField) -> Option<Vec<Assist>> {
    let mut fixes = if d.method_with_same_name_exists { method_fix(ctx, &d.expr) } else { None };
    if let Some(fix) = add_field_fix(ctx, d) {
        fixes.get_or_insert_with(Vec::new).push(fix);
    }
    fixes
}

fn add_field_fix(ctx: &DiagnosticsContext<'_>, d: &hir::UnresolvedField) -> Option<Assist> {
    // Get the FileRange of the invalid field access
    let root = ctx.sema.db.parse_or_expand(d.expr.file_id);
    let expr = d.expr.value.to_node(&root);

    let error_range = ctx.sema.original_range_opt(expr.syntax())?;
    // Convert the receiver to an ADT
    let adt = d.receiver.as_adt()?;
    let Adt::Struct(adt) = adt else {
        return None;
    };

    let target_module = adt.module(ctx.sema.db);

    let suggested_type =
        if let Some(new_field_type) = ctx.sema.type_of_expr(&expr).map(|v| v.adjusted()) {
            let display =
                new_field_type.display_source_code(ctx.sema.db, target_module.into(), true).ok();
            make::ty(display.as_deref().unwrap_or("()"))
        } else {
            make::ty("()")
        };

    if !is_editable_crate(target_module.krate(), ctx.sema.db) {
        return None;
    }
    let adt_source = adt.source(ctx.sema.db)?;
    let adt_syntax = adt_source.syntax();
    let range = adt_syntax.original_file_range(ctx.sema.db);

    // Get range of final field in the struct
    let (offset, needs_comma, indent) = match adt.fields(ctx.sema.db).last() {
        Some(field) => {
            let last_field = field.source(ctx.sema.db)?.value;
            let hir::FieldSource::Named(record_field) = last_field else {
                return None;
            };
            let last_field_syntax = record_field.syntax();
            let last_field_imdent = IndentLevel::from_node(last_field_syntax);
            (
                last_field_syntax.text_range().end(),
                !last_field_syntax.to_string().ends_with(','),
                last_field_imdent,
            )
        }
        None => {
            // Empty Struct. Add a field right before the closing brace
            let indent = IndentLevel::from_node(adt_syntax.value) + 1;
            let record_field_list =
                adt_syntax.value.children().find(|v| v.kind() == SyntaxKind::RECORD_FIELD_LIST)?;
            let offset = record_field_list.first_token().map(|f| f.text_range().end())?;
            (offset, false, indent)
        }
    };

    let field_name = make::name(d.name.as_str()?);

    // If the Type is in the same file. We don't need to add a visibility modifier. Otherwise make it pub(crate)
    let visibility = if error_range.file_id == range.file_id { "" } else { "pub(crate)" };
    let mut src_change_builder = SourceChangeBuilder::new(range.file_id);
    let comma = if needs_comma { "," } else { "" };
    src_change_builder
        .insert(offset, format!("{comma}\n{indent}{visibility}{field_name}: {suggested_type}\n"));

    // FIXME: Add a Snippet for the new field type
    let source_change = src_change_builder.finish();
    Some(Assist {
        id: AssistId("add-field-to-type", AssistKind::QuickFix),
        label: Label::new("Add field to type".to_owned()),
        group: None,
        target: error_range.range,
        source_change: Some(source_change),
        trigger_signature_help: false,
    })
}
// FIXME: We should fill out the call here, move the cursor and trigger signature help
fn method_fix(
    ctx: &DiagnosticsContext<'_>,
    expr_ptr: &InFile<AstPtr<ast::Expr>>,
) -> Option<Vec<Assist>> {
    let root = ctx.sema.db.parse_or_expand(expr_ptr.file_id);
    let expr = expr_ptr.value.to_node(&root);
    let FileRange { range, file_id } = ctx.sema.original_range_opt(expr.syntax())?;
    Some(vec![Assist {
        id: AssistId("expected-field-found-method-call-fix", AssistKind::QuickFix),
        label: Label::new("Use parentheses to call the method".to_owned()),
        group: None,
        target: range,
        source_change: Some(SourceChange::from_text_edit(
            file_id,
            TextEdit::insert(range.end(), "()".to_owned()),
        )),
        trigger_signature_help: false,
    }])
}
#[cfg(test)]
mod tests {
    use crate::{
        tests::{
            check_diagnostics, check_diagnostics_with_config, check_diagnostics_with_disabled,
        },
        DiagnosticsConfig,
    };

    #[test]
    fn smoke_test() {
        check_diagnostics(
            r#"
fn main() {
    ().foo;
    // ^^^ error: no field `foo` on type `()`
}
"#,
        );
    }

    #[test]
    fn method_clash() {
        check_diagnostics(
            r#"
struct Foo;
impl Foo {
    fn bar(&self) {}
}
fn foo() {
    Foo.bar;
     // ^^^ 💡 error: no field `bar` on type `Foo`, but a method with a similar name exists
}
"#,
        );
    }

    #[test]
    fn method_trait_() {
        check_diagnostics(
            r#"
struct Foo;
trait Bar {
    fn bar(&self) {}
}
impl Bar for Foo {}
fn foo() {
    Foo.bar;
     // ^^^ 💡 error: no field `bar` on type `Foo`, but a method with a similar name exists
}
"#,
        );
    }

    #[test]
    fn method_trait_2() {
        check_diagnostics(
            r#"
struct Foo;
trait Bar {
    fn bar(&self);
}
impl Bar for Foo {
    fn bar(&self) {}
}
fn foo() {
    Foo.bar;
     // ^^^ 💡 error: no field `bar` on type `Foo`, but a method with a similar name exists
}
"#,
        );
    }

    #[test]
    fn no_diagnostic_on_unknown() {
        check_diagnostics_with_disabled(
            r#"
fn foo() {
    x.foo;
    (&x).foo;
    (&((x,),),).foo;
}
"#,
            &["E0425"],
        );
    }

    #[test]
    fn no_diagnostic_for_missing_name() {
        let mut config = DiagnosticsConfig::test_sample();
        config.disabled.insert("syntax-error".to_owned());
        check_diagnostics_with_config(config, "fn foo() { (). }");
    }
}
