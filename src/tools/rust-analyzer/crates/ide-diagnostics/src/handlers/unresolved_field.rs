use std::iter;

use hir::{db::ExpandDatabase, Adt, HasSource, HirDisplay, InFile, Struct, Union};
use ide_db::{
    assists::{Assist, AssistId, AssistKind},
    base_db::FileRange,
    helpers::is_editable_crate,
    label::Label,
    source_change::{SourceChange, SourceChangeBuilder},
};
use syntax::{
    algo,
    ast::{self, edit::IndentLevel, make, FieldList, Name, Visibility},
    AstNode, AstPtr, Direction, SyntaxKind, TextSize,
};
use syntax::{
    ast::{edit::AstNodeEdit, Type},
    SyntaxNode,
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
    let mut fixes = Vec::new();
    if d.method_with_same_name_exists {
        fixes.extend(method_fix(ctx, &d.expr));
    }
    fixes.extend(field_fix(ctx, d));
    if fixes.is_empty() {
        None
    } else {
        Some(fixes)
    }
}

// FIXME: Add Snippet Support
fn field_fix(ctx: &DiagnosticsContext<'_>, d: &hir::UnresolvedField) -> Option<Assist> {
    // Get the FileRange of the invalid field access
    let root = ctx.sema.db.parse_or_expand(d.expr.file_id);
    let expr = d.expr.value.to_node(&root);

    let error_range = ctx.sema.original_range_opt(expr.syntax())?;
    let field_name = d.name.as_str()?;
    // Convert the receiver to an ADT
    let adt = d.receiver.strip_references().as_adt()?;
    let target_module = adt.module(ctx.sema.db);

    let suggested_type =
        if let Some(new_field_type) = ctx.sema.type_of_expr(&expr).map(|v| v.adjusted()) {
            let display =
                new_field_type.display_source_code(ctx.sema.db, target_module.into(), false).ok();
            make::ty(display.as_deref().unwrap_or("()"))
        } else {
            make::ty("()")
        };

    if !is_editable_crate(target_module.krate(), ctx.sema.db) {
        return None;
    }

    match adt {
        Adt::Struct(adt_struct) => {
            add_field_to_struct_fix(ctx, adt_struct, field_name, suggested_type, error_range)
        }
        Adt::Union(adt_union) => {
            add_variant_to_union(ctx, adt_union, field_name, suggested_type, error_range)
        }
        _ => None,
    }
}

fn add_variant_to_union(
    ctx: &DiagnosticsContext<'_>,
    adt_union: Union,
    field_name: &str,
    suggested_type: Type,
    error_range: FileRange,
) -> Option<Assist> {
    let adt_source = adt_union.source(ctx.sema.db)?;
    let adt_syntax = adt_source.syntax();
    let field_list = adt_source.value.record_field_list()?;
    let range = adt_syntax.original_file_range_rooted(ctx.sema.db);
    let field_name = make::name(field_name);

    let (offset, record_field) =
        record_field_layout(None, field_name, suggested_type, field_list, adt_syntax.value)?;

    let mut src_change_builder = SourceChangeBuilder::new(range.file_id);
    src_change_builder.insert(offset, record_field);
    Some(Assist {
        id: AssistId("add-variant-to-union", AssistKind::QuickFix),
        label: Label::new("Add field to union".to_owned()),
        group: None,
        target: error_range.range,
        source_change: Some(src_change_builder.finish()),
        trigger_signature_help: false,
    })
}

fn add_field_to_struct_fix(
    ctx: &DiagnosticsContext<'_>,
    adt_struct: Struct,
    field_name: &str,
    suggested_type: Type,
    error_range: FileRange,
) -> Option<Assist> {
    let struct_source = adt_struct.source(ctx.sema.db)?;
    let struct_syntax = struct_source.syntax();
    let struct_range = struct_syntax.original_file_range_rooted(ctx.sema.db);
    let field_list = struct_source.value.field_list();
    match field_list {
        Some(FieldList::RecordFieldList(field_list)) => {
            // Get range of final field in the struct
            let visibility = if error_range.file_id == struct_range.file_id {
                None
            } else {
                Some(make::visibility_pub_crate())
            };
            let field_name = make::name(field_name);

            let (offset, record_field) = record_field_layout(
                visibility,
                field_name,
                suggested_type,
                field_list,
                struct_syntax.value,
            )?;

            let mut src_change_builder = SourceChangeBuilder::new(struct_range.file_id);

            // FIXME: Allow for choosing a visibility modifier see https://github.com/rust-lang/rust-analyzer/issues/11563
            src_change_builder.insert(offset, record_field);
            Some(Assist {
                id: AssistId("add-field-to-record-struct", AssistKind::QuickFix),
                label: Label::new("Add field to Record Struct".to_owned()),
                group: None,
                target: error_range.range,
                source_change: Some(src_change_builder.finish()),
                trigger_signature_help: false,
            })
        }
        None => {
            // Add a field list to the Unit Struct
            let mut src_change_builder = SourceChangeBuilder::new(struct_range.file_id);
            let field_name = make::name(field_name);
            let visibility = if error_range.file_id == struct_range.file_id {
                None
            } else {
                Some(make::visibility_pub_crate())
            };
            // FIXME: Allow for choosing a visibility modifier see https://github.com/rust-lang/rust-analyzer/issues/11563
            let indent = IndentLevel::from_node(struct_syntax.value) + 1;

            let field = make::record_field(visibility, field_name, suggested_type).indent(indent);
            let record_field_list = make::record_field_list(iter::once(field));
            // A Unit Struct with no `;` is invalid syntax. We should not suggest this fix.
            let semi_colon =
                algo::skip_trivia_token(struct_syntax.value.last_token()?, Direction::Prev)?;
            if semi_colon.kind() != SyntaxKind::SEMICOLON {
                return None;
            }
            src_change_builder.replace(semi_colon.text_range(), record_field_list.to_string());

            Some(Assist {
                id: AssistId("convert-unit-struct-to-record-struct", AssistKind::QuickFix),
                label: Label::new("Convert Unit Struct to Record Struct and add field".to_owned()),
                group: None,
                target: error_range.range,
                source_change: Some(src_change_builder.finish()),
                trigger_signature_help: false,
            })
        }
        Some(FieldList::TupleFieldList(_tuple)) => {
            // FIXME: Add support for Tuple Structs. Tuple Structs are not sent to this diagnostic
            None
        }
    }
}

/// Used to determine the layout of the record field in the struct.
fn record_field_layout(
    visibility: Option<Visibility>,
    name: Name,
    suggested_type: Type,
    field_list: ast::RecordFieldList,
    struct_syntax: &SyntaxNode,
) -> Option<(TextSize, String)> {
    let (offset, needs_comma, trailing_new_line, indent) = match field_list.fields().last() {
        Some(record_field) => {
            let syntax = algo::skip_trivia_token(field_list.r_curly_token()?, Direction::Prev)?;

            let last_field_syntax = record_field.syntax();
            let last_field_indent = IndentLevel::from_node(last_field_syntax);
            (
                last_field_syntax.text_range().end(),
                syntax.kind() != SyntaxKind::COMMA,
                false,
                last_field_indent,
            )
        }
        // Empty Struct. Add a field right before the closing brace
        None => {
            let indent = IndentLevel::from_node(struct_syntax) + 1;
            let offset = field_list.r_curly_token()?.text_range().start();
            (offset, false, true, indent)
        }
    };
    let comma = if needs_comma { ",\n" } else { "" };
    let trailing_new_line = if trailing_new_line { "\n" } else { "" };
    let record_field = make::record_field(visibility, name, suggested_type);

    Some((offset, format!("{comma}{indent}{record_field}{trailing_new_line}")))
}

// FIXME: We should fill out the call here, move the cursor and trigger signature help
fn method_fix(
    ctx: &DiagnosticsContext<'_>,
    expr_ptr: &InFile<AstPtr<ast::Expr>>,
) -> Option<Assist> {
    let root = ctx.sema.db.parse_or_expand(expr_ptr.file_id);
    let expr = expr_ptr.value.to_node(&root);
    let FileRange { range, file_id } = ctx.sema.original_range_opt(expr.syntax())?;
    Some(Assist {
        id: AssistId("expected-field-found-method-call-fix", AssistKind::QuickFix),
        label: Label::new("Use parentheses to call the method".to_owned()),
        group: None,
        target: range,
        source_change: Some(SourceChange::from_text_edit(
            file_id,
            TextEdit::insert(range.end(), "()".to_owned()),
        )),
        trigger_signature_help: false,
    })
}
#[cfg(test)]
mod tests {

    use crate::{
        tests::{
            check_diagnostics, check_diagnostics_with_config, check_diagnostics_with_disabled,
            check_fix,
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

    #[test]
    fn unresolved_field_fix_on_unit() {
        check_fix(
            r#"
                struct Foo;

                fn foo() {
                    Foo.bar$0;
                }
            "#,
            r#"
                struct Foo{ bar: () }

                fn foo() {
                    Foo.bar;
                }
            "#,
        );
    }
    #[test]
    fn unresolved_field_fix_on_empty() {
        check_fix(
            r#"
                struct Foo{
                }

                fn foo() {
                    let foo = Foo{};
                    foo.bar$0;
                }
            "#,
            r#"
                struct Foo{
                    bar: ()
                }

                fn foo() {
                    let foo = Foo{};
                    foo.bar;
                }
            "#,
        );
    }
    #[test]
    fn unresolved_field_fix_on_struct() {
        check_fix(
            r#"
                struct Foo{
                    a: i32
                }

                fn foo() {
                    let foo = Foo{a: 0};
                    foo.bar$0;
                }
            "#,
            r#"
                struct Foo{
                    a: i32,
                    bar: ()
                }

                fn foo() {
                    let foo = Foo{a: 0};
                    foo.bar;
                }
            "#,
        );
    }
    #[test]
    fn unresolved_field_fix_on_union() {
        check_fix(
            r#"
                union Foo{
                    a: i32
                }

                fn foo() {
                    let foo = Foo{a: 0};
                    foo.bar$0;
                }
            "#,
            r#"
                union Foo{
                    a: i32,
                    bar: ()
                }

                fn foo() {
                    let foo = Foo{a: 0};
                    foo.bar;
                }
            "#,
        );
    }
}
