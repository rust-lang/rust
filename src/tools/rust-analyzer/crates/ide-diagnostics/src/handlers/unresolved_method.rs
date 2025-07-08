use hir::{FileRange, HirDisplay, InFile, db::ExpandDatabase};
use ide_db::text_edit::TextEdit;
use ide_db::{
    assists::{Assist, AssistId},
    label::Label,
    source_change::SourceChange,
};
use syntax::{
    AstNode, SmolStr, TextRange, ToSmolStr,
    ast::{self, HasArgList, make},
    format_smolstr,
};

use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext, adjusted_display_range};

// Diagnostic: unresolved-method
//
// This diagnostic is triggered if a method does not exist on a given type.
pub(crate) fn unresolved_method(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::UnresolvedMethodCall<'_>,
) -> Diagnostic {
    let suffix = if d.field_with_same_name.is_some() {
        ", but a field with a similar name exists"
    } else if d.assoc_func_with_same_name.is_some() {
        ", but an associated function with a similar name exists"
    } else {
        ""
    };
    Diagnostic::new(
        DiagnosticCode::RustcHardError("E0599"),
        format!(
            "no method `{}` on type `{}`{suffix}",
            d.name.display(ctx.sema.db, ctx.edition),
            d.receiver.display(ctx.sema.db, ctx.display_target)
        ),
        adjusted_display_range(ctx, d.expr, &|expr| {
            Some(
                match expr.left()? {
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
}

fn fixes(ctx: &DiagnosticsContext<'_>, d: &hir::UnresolvedMethodCall<'_>) -> Option<Vec<Assist>> {
    let field_fix = if let Some(ty) = &d.field_with_same_name {
        field_fix(ctx, d, ty)
    } else {
        // FIXME: add quickfix
        None
    };

    let assoc_func_fix = assoc_func_fix(ctx, d);

    let mut fixes = vec![];
    if let Some(field_fix) = field_fix {
        fixes.push(field_fix);
    }
    if let Some(assoc_func_fix) = assoc_func_fix {
        fixes.push(assoc_func_fix);
    }

    if fixes.is_empty() { None } else { Some(fixes) }
}

fn field_fix(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::UnresolvedMethodCall<'_>,
    ty: &hir::Type<'_>,
) -> Option<Assist> {
    if !ty.impls_fnonce(ctx.sema.db) {
        return None;
    }
    let expr_ptr = &d.expr;
    let root = ctx.sema.db.parse_or_expand(expr_ptr.file_id);
    let expr = expr_ptr.value.to_node(&root);
    let (file_id, range) = match expr.left()? {
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
    Some(Assist {
        id: AssistId::quick_fix("expected-method-found-field-fix"),
        label: Label::new("Use parentheses to call the value of the field".to_owned()),
        group: None,
        target: range,
        source_change: Some(SourceChange::from_iter([
            (file_id.file_id(ctx.sema.db), TextEdit::insert(range.start(), "(".to_owned())),
            (file_id.file_id(ctx.sema.db), TextEdit::insert(range.end(), ")".to_owned())),
        ])),
        command: None,
    })
}

fn assoc_func_fix(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::UnresolvedMethodCall<'_>,
) -> Option<Assist> {
    if let Some(f) = d.assoc_func_with_same_name {
        let db = ctx.sema.db;

        let expr_ptr = &d.expr;
        let root = db.parse_or_expand(expr_ptr.file_id);
        let expr: ast::Expr = expr_ptr.value.to_node(&root).left()?;

        let call = ast::MethodCallExpr::cast(expr.syntax().clone())?;
        let range = InFile::new(expr_ptr.file_id, call.syntax().text_range())
            .original_node_file_range_rooted_opt(db)?;

        let receiver = call.receiver()?;
        let receiver_type = &ctx.sema.type_of_expr(&receiver)?.original;

        let assoc_fn_params = f.assoc_fn_params(db);
        let need_to_take_receiver_as_first_arg = if assoc_fn_params.is_empty() {
            false
        } else {
            assoc_fn_params
                .first()
                .map(|first_arg| {
                    // For generic type, say `Box`, take `Box::into_raw(b: Self)` as example,
                    // type of `b` is `Self`, which is `Box<T, A>`, containing unspecified generics.
                    // However, type of `receiver` is specified, it could be `Box<i32, Global>` or something like that,
                    // so `first_arg.ty() == receiver_type` evaluate to `false` here.
                    // Here add `first_arg.ty().as_adt() == receiver_type.as_adt()` as guard,
                    // apply `.as_adt()` over `Box<T, A>` or `Box<i32, Global>` gets `Box`, so we get `true` here.

                    // FIXME: it fails when type of `b` is `Box` with other generic param different from `receiver`
                    first_arg.ty() == receiver_type
                        || first_arg.ty().as_adt() == receiver_type.as_adt()
                })
                .unwrap_or(false)
        };

        let mut receiver_type_adt_name =
            receiver_type.as_adt()?.name(db).display_no_db(ctx.edition).to_smolstr();

        let generic_parameters: Vec<SmolStr> =
            receiver_type.generic_parameters(db, ctx.display_target).collect();
        // if receiver should be pass as first arg in the assoc func,
        // we could omit generic parameters cause compiler can deduce it automatically
        if !need_to_take_receiver_as_first_arg && !generic_parameters.is_empty() {
            let generic_parameters = generic_parameters.join(", ");
            receiver_type_adt_name =
                format_smolstr!("{receiver_type_adt_name}::<{generic_parameters}>");
        }

        let method_name = call.name_ref()?;
        let assoc_func_path = format!("{receiver_type_adt_name}::{method_name}");

        let assoc_func_path = make::expr_path(make::path_from_text(&assoc_func_path));

        let args: Vec<_> = if need_to_take_receiver_as_first_arg {
            std::iter::once(receiver).chain(call.arg_list()?.args()).collect()
        } else {
            call.arg_list()?.args().collect()
        };
        let args = make::arg_list(args);

        let assoc_func_call_expr_string = make::expr_call(assoc_func_path, args).to_string();

        Some(Assist {
            id: AssistId::quick_fix("method_call_to_assoc_func_call_fix"),
            label: Label::new(format!(
                "Use associated func call instead: `{assoc_func_call_expr_string}`"
            )),
            group: None,
            target: range.range,
            source_change: Some(SourceChange::from_text_edit(
                range.file_id.file_id(ctx.sema.db),
                TextEdit::replace(range.range, assoc_func_call_expr_string),
            )),
            command: None,
        })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_diagnostics, check_fix};

    #[test]
    fn test_assoc_func_fix() {
        check_fix(
            r#"
struct A {}

impl A {
    fn hello() {}
}
fn main() {
    let a = A{};
    a.hello$0();
}
"#,
            r#"
struct A {}

impl A {
    fn hello() {}
}
fn main() {
    let a = A{};
    A::hello();
}
"#,
        );
    }

    #[test]
    fn test_assoc_func_diagnostic() {
        check_diagnostics(
            r#"
struct A {}
impl A {
    fn hello() {}
}
fn main() {
    let a = A{};
    a.hello();
   // ^^^^^ 💡 error: no method `hello` on type `A`, but an associated function with a similar name exists
}
"#,
        );
    }

    #[test]
    fn test_assoc_func_fix_with_generic() {
        check_fix(
            r#"
struct A<T, U> {
    a: T,
    b: U
}

impl<T, U> A<T, U> {
    fn foo() {}
}
fn main() {
    let a = A {a: 0, b: ""};
    a.foo()$0;
}
"#,
            r#"
struct A<T, U> {
    a: T,
    b: U
}

impl<T, U> A<T, U> {
    fn foo() {}
}
fn main() {
    let a = A {a: 0, b: ""};
    A::<i32, &'static str>::foo();
}
"#,
        );
    }

    #[test]
    fn smoke_test() {
        check_diagnostics(
            r#"
fn main() {
    ().foo();
    // ^^^ error: no method `foo` on type `()`
}
"#,
        );
    }

    #[test]
    fn smoke_test_in_macro_def_site() {
        check_diagnostics(
            r#"
macro_rules! m {
    ($rcv:expr) => {
        $rcv.foo()
    }
}
fn main() {
    m!(());
 // ^^ error: no method `foo` on type `()`
}
"#,
        );
    }

    #[test]
    fn smoke_test_in_macro_call_site() {
        check_diagnostics(
            r#"
macro_rules! m {
    ($ident:ident) => {
        ().$ident()
    }
}
fn main() {
    m!(foo);
    // ^^^ error: no method `foo` on type `()`
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
    Foo { bar: 0 }.bar();
                // ^^^ error: no method `bar` on type `Foo`, but a field with a similar name exists
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
