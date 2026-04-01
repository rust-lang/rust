use either::Either;
use hir::Semantics;
use ide_db::text_edit::TextEdit;
use ide_db::ty_filter::TryEnum;
use ide_db::{RootDatabase, source_change::SourceChange};
use syntax::{AstNode, ast};

use crate::{Assist, Diagnostic, DiagnosticCode, DiagnosticsContext, fix};

// Diagnostic: non-exhaustive-let
//
// This diagnostic is triggered if a `let` statement without an `else` branch has a non-exhaustive
// pattern.
pub(crate) fn non_exhaustive_let(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::NonExhaustiveLet,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0005"),
        format!("non-exhaustive pattern: {}", d.uncovered_patterns),
        d.pat.map(Into::into),
    )
    .stable()
    .with_fixes(fixes(&ctx.sema, d))
}

fn fixes(sema: &Semantics<'_, RootDatabase>, d: &hir::NonExhaustiveLet) -> Option<Vec<Assist>> {
    let root = sema.parse_or_expand(d.pat.file_id);
    let pat = d.pat.value.to_node(&root);
    let let_stmt = ast::LetStmt::cast(pat.syntax().parent()?)?;
    let early_node =
        sema.ancestors_with_macros(let_stmt.syntax().clone()).find_map(AstNode::cast)?;
    let early_text = early_text(sema, &early_node);

    if let_stmt.let_else().is_some() {
        return None;
    }
    let hir::FileRangeWrapper { file_id, range } = sema.original_range_opt(let_stmt.syntax())?;
    let insert_offset = if let Some(semicolon) = let_stmt.semicolon_token()
        && let Some(token) = sema.parse(file_id).syntax().token_at_offset(range.end()).left_biased()
        && token.kind() == semicolon.kind()
    {
        token.text_range().start()
    } else {
        range.end()
    };
    let semicolon = if let_stmt.semicolon_token().is_none() { ";" } else { "" };
    let else_block = format!(" else {{ {early_text} }}{semicolon}");
    let file_id = file_id.file_id(sema.db);

    let source_change =
        SourceChange::from_text_edit(file_id, TextEdit::insert(insert_offset, else_block));
    let target = sema.original_range(let_stmt.syntax()).range;
    Some(vec![fix("add_let_else_block", "Add let-else block", source_change, target)])
}

fn early_text(
    sema: &Semantics<'_, RootDatabase>,
    early_node: &Either<ast::AnyHasLoopBody, Either<ast::Fn, ast::ClosureExpr>>,
) -> &'static str {
    match early_node {
        Either::Left(_any_loop) => "continue",
        Either::Right(Either::Left(fn_)) => sema
            .to_def(fn_)
            .map(|fn_def| fn_def.ret_type(sema.db))
            .map(|ty| return_text(&ty, sema))
            .unwrap_or("return"),
        Either::Right(Either::Right(closure)) => closure
            .body()
            .and_then(|expr| sema.type_of_expr(&expr))
            .map(|ty| return_text(&ty.adjusted(), sema))
            .unwrap_or("return"),
    }
}

fn return_text(ty: &hir::Type<'_>, sema: &Semantics<'_, RootDatabase>) -> &'static str {
    if ty.is_unit() {
        "return"
    } else if let Some(try_enum) = TryEnum::from_ty(sema, ty) {
        match try_enum {
            TryEnum::Option => "return None",
            TryEnum::Result => "return Err($0)",
        }
    } else {
        "return $0"
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_diagnostics, check_fix};

    #[test]
    fn option_nonexhaustive() {
        check_diagnostics(
            r#"
//- minicore: option
fn main() {
    let None = Some(5);
      //^^^^ 💡 error: non-exhaustive pattern: `Some(_)` not covered
}
"#,
        );
    }

    #[test]
    fn option_exhaustive() {
        check_diagnostics(
            r#"
//- minicore: option
fn main() {
    let Some(_) | None = Some(5);
}
"#,
        );
    }

    #[test]
    fn option_nonexhaustive_inside_blocks() {
        check_diagnostics(
            r#"
//- minicore: option
fn main() {
    '_a: {
        let None = Some(5);
          //^^^^ 💡 error: non-exhaustive pattern: `Some(_)` not covered
    }
}
"#,
        );

        check_diagnostics(
            r#"
//- minicore: future, option
fn main() {
    let _ = async {
        let None = Some(5);
          //^^^^ 💡 error: non-exhaustive pattern: `Some(_)` not covered
    };
}
"#,
        );

        check_diagnostics(
            r#"
//- minicore: option
fn main() {
    unsafe {
        let None = Some(5);
          //^^^^ 💡 error: non-exhaustive pattern: `Some(_)` not covered
    }
}
"#,
        );
    }

    #[test]
    fn min_exhaustive() {
        check_diagnostics(
            r#"
//- minicore: result
fn test(x: Result<i32, !>) {
    let Ok(_y) = x;
}
"#,
        );

        check_diagnostics(
            r#"
//- minicore: result
fn test(x: Result<i32, &'static !>) {
    let Ok(_y) = x;
      //^^^^^^ 💡 error: non-exhaustive pattern: `Err(_)` not covered
}
"#,
        );
    }

    #[test]
    fn empty_patterns_normalize() {
        check_diagnostics(
            r#"
enum Infallible {}

trait Foo {
    type Assoc;
}
enum Enum<T: Foo> {
    A,
    B(T::Assoc),
}

impl Foo for () {
    type Assoc = Infallible;
}

fn foo(v: Enum<()>) {
    let Enum::A = v;
}
        "#,
        );
    }

    #[test]
    fn fix_return_in_loop() {
        check_fix(
            r#"
//- minicore: option
fn foo() {
    while cond {
        let None$0 = Some(5);
    }
}
"#,
            r#"
fn foo() {
    while cond {
        let None = Some(5) else { continue };
    }
}
"#,
        );
    }

    #[test]
    fn fix_return_in_fn() {
        check_fix(
            r#"
//- minicore: option
fn foo() {
    let None$0 = Some(5);
}
"#,
            r#"
fn foo() {
    let None = Some(5) else { return };
}
"#,
        );
    }

    #[test]
    fn fix_return_in_macro_expanded() {
        check_fix(
            r#"
//- minicore: option
macro_rules! identity { ($($t:tt)*) => { $($t)* }; }
fn foo() {
    identity! {
        let None$0 = Some(5);
    }
}
"#,
            r#"
macro_rules! identity { ($($t:tt)*) => { $($t)* }; }
fn foo() {
    identity! {
        let None = Some(5) else { return };
    }
}
"#,
        );
    }

    #[test]
    fn fix_return_in_incomplete_let() {
        check_fix(
            r#"
//- minicore: option
fn foo() {
    let None$0 = Some(5)
}
"#,
            r#"
fn foo() {
    let None = Some(5) else { return };
}
"#,
        );
    }

    #[test]
    fn fix_return_in_closure() {
        check_fix(
            r#"
//- minicore: option
fn foo() -> Option<()> {
    let _f = || {
        let None$0 = Some(5);
    };
}
"#,
            r#"
fn foo() -> Option<()> {
    let _f = || {
        let None = Some(5) else { return };
    };
}
"#,
        );
    }

    #[test]
    fn fix_return_try_in_fn() {
        check_fix(
            r#"
//- minicore: option
fn foo() -> Option<()> {
    let None$0 = Some(5);
}
"#,
            r#"
fn foo() -> Option<()> {
    let None = Some(5) else { return None };
}
"#,
        );

        check_fix(
            r#"
//- minicore: option, result
fn foo() -> Result<(), i32> {
    let None$0 = Some(5);
}
"#,
            r#"
fn foo() -> Result<(), i32> {
    let None = Some(5) else { return Err($0) };
}
"#,
        );
    }

    #[test]
    fn regression_20259() {
        check_diagnostics(
            r#"
//- minicore: deref
use core::ops::Deref;

struct Foo<T>(T);

impl<T> Deref for Foo<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

fn test(x: Foo<(i32, bool)>) {
    let (_a, _b): &(i32, bool) = &x;
}
"#,
        );
    }

    #[test]
    fn uninhabited_variants() {
        check_diagnostics(
            r#"
//- minicore: result
enum Infallible {}

trait Foo {
    type Bar;
}

struct Wrapper<T> {
    error: T,
}

struct FooWrapper<T: Foo> {
    error: T::Bar,
}

fn foo<T: Foo<Bar = Infallible>>(result: Result<T, T::Bar>) -> T {
    let Ok(ok) = result;
    ok
}

fn bar<T: Foo<Bar = Infallible>>(result: Result<T, (T::Bar,)>) -> T {
    let Ok(ok) = result;
    ok
}

fn baz<T: Foo<Bar = Infallible>>(result: Result<T, Wrapper<T::Bar>>) -> T {
    let Ok(ok) = result;
    ok
}

fn qux<T: Foo<Bar = Infallible>>(result: Result<T, FooWrapper<T>>) -> T {
    let Ok(ok) = result;
    ok
}

fn quux<T: Foo<Bar = Infallible>>(result: Result<T, [T::Bar; 1]>) -> T {
    let Ok(ok) = result;
    ok
}

fn corge<T: Foo<Bar = Infallible>>(result: Result<T, (i32, T::Bar)>) -> T {
    let Ok(ok) = result;
    ok
}
"#,
        );
    }
}
