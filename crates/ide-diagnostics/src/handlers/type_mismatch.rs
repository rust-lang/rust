use either::Either;
use hir::{db::AstDatabase, HirDisplay, InFile, Type};
use ide_db::{famous_defs::FamousDefs, source_change::SourceChange};
use syntax::{
    ast::{self, BlockExpr, ExprStmt},
    AstNode, AstPtr,
};
use text_edit::TextEdit;

use crate::{adjusted_display_range, fix, Assist, Diagnostic, DiagnosticsContext};

// Diagnostic: type-mismatch
//
// This diagnostic is triggered when the type of an expression or pattern does not match
// the expected type.
pub(crate) fn type_mismatch(ctx: &DiagnosticsContext<'_>, d: &hir::TypeMismatch) -> Diagnostic {
    let display_range = match &d.expr_or_pat {
        Either::Left(expr) => adjusted_display_range::<ast::BlockExpr>(
            ctx,
            expr.clone().map(|it| it.into()),
            &|block| {
                let r_curly_range = block.stmt_list()?.r_curly_token()?.text_range();
                cov_mark::hit!(type_mismatch_on_block);
                Some(r_curly_range)
            },
        ),
        Either::Right(pat) => {
            ctx.sema.diagnostics_display_range(pat.clone().map(|it| it.into())).range
        }
    };
    let mut diag = Diagnostic::new(
        "type-mismatch",
        format!(
            "expected {}, found {}",
            d.expected.display(ctx.sema.db),
            d.actual.display(ctx.sema.db)
        ),
        display_range,
    )
    .with_fixes(fixes(ctx, d));
    if diag.fixes.is_none() {
        diag.experimental = true;
    }
    diag
}

fn fixes(ctx: &DiagnosticsContext<'_>, d: &hir::TypeMismatch) -> Option<Vec<Assist>> {
    let mut fixes = Vec::new();

    match &d.expr_or_pat {
        Either::Left(expr_ptr) => {
            add_reference(ctx, d, expr_ptr, &mut fixes);
            add_missing_ok_or_some(ctx, d, expr_ptr, &mut fixes);
            remove_semicolon(ctx, d, expr_ptr, &mut fixes);
            str_ref_to_owned(ctx, d, expr_ptr, &mut fixes);
        }
        Either::Right(_pat_ptr) => {}
    }

    if fixes.is_empty() {
        None
    } else {
        Some(fixes)
    }
}

fn add_reference(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::TypeMismatch,
    expr_ptr: &InFile<AstPtr<ast::Expr>>,
    acc: &mut Vec<Assist>,
) -> Option<()> {
    let range = ctx.sema.diagnostics_display_range(expr_ptr.clone().map(|it| it.into())).range;

    let (_, mutability) = d.expected.as_reference()?;
    let actual_with_ref = Type::reference(&d.actual, mutability);
    if !actual_with_ref.could_coerce_to(ctx.sema.db, &d.expected) {
        return None;
    }

    let ampersands = format!("&{}", mutability.as_keyword_for_ref());

    let edit = TextEdit::insert(range.start(), ampersands);
    let source_change =
        SourceChange::from_text_edit(expr_ptr.file_id.original_file(ctx.sema.db), edit);
    acc.push(fix("add_reference_here", "Add reference here", source_change, range));
    Some(())
}

fn add_missing_ok_or_some(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::TypeMismatch,
    expr_ptr: &InFile<AstPtr<ast::Expr>>,
    acc: &mut Vec<Assist>,
) -> Option<()> {
    let root = ctx.sema.db.parse_or_expand(expr_ptr.file_id)?;
    let expr = expr_ptr.value.to_node(&root);
    let expr_range = expr.syntax().text_range();
    let scope = ctx.sema.scope(expr.syntax())?;

    let expected_adt = d.expected.as_adt()?;
    let expected_enum = expected_adt.as_enum()?;

    let famous_defs = FamousDefs(&ctx.sema, scope.krate());
    let core_result = famous_defs.core_result_Result();
    let core_option = famous_defs.core_option_Option();

    if Some(expected_enum) != core_result && Some(expected_enum) != core_option {
        return None;
    }

    let variant_name = if Some(expected_enum) == core_result { "Ok" } else { "Some" };

    let wrapped_actual_ty = expected_adt.ty_with_args(ctx.sema.db, &[d.actual.clone()]);

    if !d.expected.could_unify_with(ctx.sema.db, &wrapped_actual_ty) {
        return None;
    }

    let mut builder = TextEdit::builder();
    builder.insert(expr.syntax().text_range().start(), format!("{variant_name}("));
    builder.insert(expr.syntax().text_range().end(), ")".to_string());
    let source_change =
        SourceChange::from_text_edit(expr_ptr.file_id.original_file(ctx.sema.db), builder.finish());
    let name = format!("Wrap in {variant_name}");
    acc.push(fix("wrap_in_constructor", &name, source_change, expr_range));
    Some(())
}

fn remove_semicolon(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::TypeMismatch,
    expr_ptr: &InFile<AstPtr<ast::Expr>>,
    acc: &mut Vec<Assist>,
) -> Option<()> {
    let root = ctx.sema.db.parse_or_expand(expr_ptr.file_id)?;
    let expr = expr_ptr.value.to_node(&root);
    if !d.actual.is_unit() {
        return None;
    }
    let block = BlockExpr::cast(expr.syntax().clone())?;
    let expr_before_semi =
        block.statements().last().and_then(|s| ExprStmt::cast(s.syntax().clone()))?;
    let type_before_semi = ctx.sema.type_of_expr(&expr_before_semi.expr()?)?.original();
    if !type_before_semi.could_coerce_to(ctx.sema.db, &d.expected) {
        return None;
    }
    let semicolon_range = expr_before_semi.semicolon_token()?.text_range();

    let edit = TextEdit::delete(semicolon_range);
    let source_change =
        SourceChange::from_text_edit(expr_ptr.file_id.original_file(ctx.sema.db), edit);

    acc.push(fix("remove_semicolon", "Remove this semicolon", source_change, semicolon_range));
    Some(())
}

fn str_ref_to_owned(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::TypeMismatch,
    expr_ptr: &InFile<AstPtr<ast::Expr>>,
    acc: &mut Vec<Assist>,
) -> Option<()> {
    let expected = d.expected.display(ctx.sema.db);
    let actual = d.actual.display(ctx.sema.db);

    // FIXME do this properly
    if expected.to_string() != "String" || actual.to_string() != "&str" {
        return None;
    }

    let root = ctx.sema.db.parse_or_expand(expr_ptr.file_id)?;
    let expr = expr_ptr.value.to_node(&root);
    let expr_range = expr.syntax().text_range();

    let to_owned = format!(".to_owned()");

    let edit = TextEdit::insert(expr.syntax().text_range().end(), to_owned);
    let source_change =
        SourceChange::from_text_edit(expr_ptr.file_id.original_file(ctx.sema.db), edit);
    acc.push(fix("str_ref_to_owned", "Add .to_owned() here", source_change, expr_range));

    Some(())
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_diagnostics, check_fix, check_no_fix};

    #[test]
    fn missing_reference() {
        check_diagnostics(
            r#"
fn main() {
    test(123);
       //^^^ 💡 error: expected &i32, found i32
}
fn test(arg: &i32) {}
"#,
        );
    }

    #[test]
    fn test_add_reference_to_int() {
        check_fix(
            r#"
fn main() {
    test(123$0);
}
fn test(arg: &i32) {}
            "#,
            r#"
fn main() {
    test(&123);
}
fn test(arg: &i32) {}
            "#,
        );
    }

    #[test]
    fn test_add_mutable_reference_to_int() {
        check_fix(
            r#"
fn main() {
    test($0123);
}
fn test(arg: &mut i32) {}
            "#,
            r#"
fn main() {
    test(&mut 123);
}
fn test(arg: &mut i32) {}
            "#,
        );
    }

    #[test]
    fn test_add_reference_to_array() {
        check_fix(
            r#"
//- minicore: coerce_unsized
fn main() {
    test($0[1, 2, 3]);
}
fn test(arg: &[i32]) {}
            "#,
            r#"
fn main() {
    test(&[1, 2, 3]);
}
fn test(arg: &[i32]) {}
            "#,
        );
    }

    #[test]
    fn test_add_reference_with_autoderef() {
        check_fix(
            r#"
//- minicore: coerce_unsized, deref
struct Foo;
struct Bar;
impl core::ops::Deref for Foo {
    type Target = Bar;
}

fn main() {
    test($0Foo);
}
fn test(arg: &Bar) {}
            "#,
            r#"
struct Foo;
struct Bar;
impl core::ops::Deref for Foo {
    type Target = Bar;
}

fn main() {
    test(&Foo);
}
fn test(arg: &Bar) {}
            "#,
        );
    }

    #[test]
    fn test_add_reference_to_method_call() {
        check_fix(
            r#"
fn main() {
    Test.call_by_ref($0123);
}
struct Test;
impl Test {
    fn call_by_ref(&self, arg: &i32) {}
}
            "#,
            r#"
fn main() {
    Test.call_by_ref(&123);
}
struct Test;
impl Test {
    fn call_by_ref(&self, arg: &i32) {}
}
            "#,
        );
    }

    #[test]
    fn test_add_reference_to_let_stmt() {
        check_fix(
            r#"
fn main() {
    let test: &i32 = $0123;
}
            "#,
            r#"
fn main() {
    let test: &i32 = &123;
}
            "#,
        );
    }

    #[test]
    fn test_add_reference_to_macro_call() {
        check_fix(
            r#"
macro_rules! thousand {
    () => {
        1000_u64
    };
}
fn test(foo: &u64) {}
fn main() {
    test($0thousand!());
}
            "#,
            r#"
macro_rules! thousand {
    () => {
        1000_u64
    };
}
fn test(foo: &u64) {}
fn main() {
    test(&thousand!());
}
            "#,
        );
    }

    #[test]
    fn test_add_mutable_reference_to_let_stmt() {
        check_fix(
            r#"
fn main() {
    let test: &mut i32 = $0123;
}
            "#,
            r#"
fn main() {
    let test: &mut i32 = &mut 123;
}
            "#,
        );
    }

    #[test]
    fn test_wrap_return_type_option() {
        check_fix(
            r#"
//- minicore: option, result
fn div(x: i32, y: i32) -> Option<i32> {
    if y == 0 {
        return None;
    }
    x / y$0
}
"#,
            r#"
fn div(x: i32, y: i32) -> Option<i32> {
    if y == 0 {
        return None;
    }
    Some(x / y)
}
"#,
        );
    }

    #[test]
    fn const_generic_type_mismatch() {
        check_diagnostics(
            r#"
            pub struct Rate<const N: u32>;
            fn f<const N: u64>() -> Rate<N> { // FIXME: add some error
                loop {}
            }
            fn run(t: Rate<5>) {
            }
            fn main() {
                run(f()) // FIXME: remove this error
                  //^^^ error: expected Rate<5>, found Rate<_>
            }
"#,
        );
    }

    #[test]
    fn const_generic_unknown() {
        check_diagnostics(
            r#"
            pub struct Rate<T, const NOM: u32, const DENOM: u32>(T);
            fn run(t: Rate<u32, 1, 1>) {
            }
            fn main() {
                run(Rate::<_, _, _>(5));
            }
"#,
        );
    }

    #[test]
    fn test_wrap_return_type_option_tails() {
        check_fix(
            r#"
//- minicore: option, result
fn div(x: i32, y: i32) -> Option<i32> {
    if y == 0 {
        Some(0)
    } else if true {
        100$0
    } else {
        None
    }
}
"#,
            r#"
fn div(x: i32, y: i32) -> Option<i32> {
    if y == 0 {
        Some(0)
    } else if true {
        Some(100)
    } else {
        None
    }
}
"#,
        );
    }

    #[test]
    fn test_wrap_return_type() {
        check_fix(
            r#"
//- minicore: option, result
fn div(x: i32, y: i32) -> Result<i32, ()> {
    if y == 0 {
        return Err(());
    }
    x / y$0
}
"#,
            r#"
fn div(x: i32, y: i32) -> Result<i32, ()> {
    if y == 0 {
        return Err(());
    }
    Ok(x / y)
}
"#,
        );
    }

    #[test]
    fn test_wrap_return_type_handles_generic_functions() {
        check_fix(
            r#"
//- minicore: option, result
fn div<T>(x: T) -> Result<T, i32> {
    if x == 0 {
        return Err(7);
    }
    $0x
}
"#,
            r#"
fn div<T>(x: T) -> Result<T, i32> {
    if x == 0 {
        return Err(7);
    }
    Ok(x)
}
"#,
        );
    }

    #[test]
    fn test_wrap_return_type_handles_type_aliases() {
        check_fix(
            r#"
//- minicore: option, result
type MyResult<T> = Result<T, ()>;

fn div(x: i32, y: i32) -> MyResult<i32> {
    if y == 0 {
        return Err(());
    }
    x $0/ y
}
"#,
            r#"
type MyResult<T> = Result<T, ()>;

fn div(x: i32, y: i32) -> MyResult<i32> {
    if y == 0 {
        return Err(());
    }
    Ok(x / y)
}
"#,
        );
    }

    #[test]
    fn test_in_const_and_static() {
        check_fix(
            r#"
//- minicore: option, result
static A: Option<()> = {($0)};
            "#,
            r#"
static A: Option<()> = {Some(())};
            "#,
        );
        check_fix(
            r#"
//- minicore: option, result
const _: Option<()> = {($0)};
            "#,
            r#"
const _: Option<()> = {Some(())};
            "#,
        );
    }

    #[test]
    fn test_wrap_return_type_not_applicable_when_expr_type_does_not_match_ok_type() {
        check_no_fix(
            r#"
//- minicore: option, result
fn foo() -> Result<(), i32> { 0$0 }
"#,
        );
    }

    #[test]
    fn test_wrap_return_type_not_applicable_when_return_type_is_not_result_or_option() {
        check_no_fix(
            r#"
//- minicore: option, result
enum SomeOtherEnum { Ok(i32), Err(String) }

fn foo() -> SomeOtherEnum { 0$0 }
"#,
        );
    }

    #[test]
    fn remove_semicolon() {
        check_fix(r#"fn f() -> i32 { 92$0; }"#, r#"fn f() -> i32 { 92 }"#);
    }

    #[test]
    fn str_ref_to_owned() {
        check_fix(
            r#"
struct String;

fn test() -> String {
    "a"$0
}
            "#,
            r#"
struct String;

fn test() -> String {
    "a".to_owned()
}
            "#,
        );
    }

    #[test]
    fn type_mismatch_on_block() {
        cov_mark::check!(type_mismatch_on_block);
        check_diagnostics(
            r#"
fn f() -> i32 {
    let x = 1;
    let y = 2;
    let _ = x + y;
  }
//^ error: expected i32, found ()
"#,
        );
    }

    #[test]
    fn type_mismatch_pat_smoke_test() {
        check_diagnostics(
            r#"
fn f() {
    let &() = &mut ();
    match &() {
        &9 => ()
       //^ error: expected (), found i32
    }
}
"#,
        );
    }
}
