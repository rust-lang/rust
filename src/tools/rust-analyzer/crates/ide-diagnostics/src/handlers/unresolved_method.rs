use hir::{db::AstDatabase, HirDisplay};
use ide_db::{
    assists::{Assist, AssistId, AssistKind},
    base_db::FileRange,
    label::Label,
    source_change::SourceChange,
};
use syntax::{ast, AstNode, TextRange};
use text_edit::TextEdit;

use crate::{Diagnostic, DiagnosticsContext};

// Diagnostic: unresolved-method
//
// This diagnostic is triggered if a method does not exist on a given type.
pub(crate) fn unresolved_method(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::UnresolvedMethodCall,
) -> Diagnostic {
    let field_suffix = if d.field_with_same_name.is_some() {
        ", but a field with a similar name exists"
    } else {
        ""
    };
    Diagnostic::new(
        "unresolved-method",
        format!(
            "no method `{}` on type `{}`{field_suffix}",
            d.name,
            d.receiver.display(ctx.sema.db)
        ),
        ctx.sema.diagnostics_display_range(d.expr.clone().map(|it| it.into())).range,
    )
    .with_fixes(fixes(ctx, d))
    .experimental()
}

fn fixes(ctx: &DiagnosticsContext<'_>, d: &hir::UnresolvedMethodCall) -> Option<Vec<Assist>> {
    if let Some(ty) = &d.field_with_same_name {
        field_fix(ctx, d, ty)
    } else {
        // FIXME: add quickfix
        None
    }
}

fn field_fix(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::UnresolvedMethodCall,
    ty: &hir::Type,
) -> Option<Vec<Assist>> {
    if !ty.impls_fnonce(ctx.sema.db) {
        return None;
    }
    let expr_ptr = &d.expr;
    let root = ctx.sema.db.parse_or_expand(expr_ptr.file_id)?;
    let expr = expr_ptr.value.to_node(&root);
    let (file_id, range) = match expr {
        ast::Expr::MethodCallExpr(mcall) => {
            let FileRange { range, file_id } =
                ctx.sema.original_range_opt(mcall.receiver()?.syntax())?;
            let FileRange { range: range2, file_id: file_id2 } =
                ctx.sema.original_range_opt(mcall.name_ref()?.syntax())?;
            if file_id != file_id2 {
                return None;
            }
            (file_id, TextRange::new(range.start(), range2.end()))
        }
        _ => return None,
    };
    Some(vec![Assist {
        id: AssistId("expected-method-found-field-fix", AssistKind::QuickFix),
        label: Label::new("Use parentheses to call the value of the field".to_string()),
        group: None,
        target: range,
        source_change: Some(SourceChange::from_iter([
            (file_id, TextEdit::insert(range.start(), "(".to_owned())),
            (file_id, TextEdit::insert(range.end(), ")".to_owned())),
        ])),
        trigger_signature_help: false,
    }])
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_diagnostics, check_fix};

    #[test]
    fn smoke_test() {
        check_diagnostics(
            r#"
fn main() {
    ().foo();
 // ^^^^^^^^ error: no method `foo` on type `()`
}
"#,
        );
    }

    #[test]
    fn field() {
        check_diagnostics(
            r#"
struct Foo { bar: i32 }
fn foo() {
    Foo { bar: i32 }.bar();
 // ^^^^^^^^^^^^^^^^^^^^^^ error: no method `bar` on type `Foo`, but a field with a similar name exists
}
"#,
        );
    }

    #[test]
    fn callable_field() {
        check_fix(
            r#"
//- minicore: fn
struct Foo { bar: fn() }
fn foo() {
    Foo { bar: foo }.b$0ar();
}
"#,
            r#"
struct Foo { bar: fn() }
fn foo() {
    (Foo { bar: foo }.bar)();
}
"#,
        );
    }
}
