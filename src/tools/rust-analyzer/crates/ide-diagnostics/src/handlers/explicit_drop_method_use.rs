use either::Either;
use hir::InFile;
use ide_db::assists::Assist;
use ide_db::source_change::{SourceChange, SourceChangeBuilder};
use ide_db::text_edit::TextEdit;
use itertools::Itertools;
use syntax::{
    AstNode, AstPtr,
    ast::{self, HasArgList},
};

use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext, adjusted_display_range, fix};

// Diagnostic: explicit-drop-method-use
//
// This diagnostic is triggered when the `Drop::drop` method is called (or named) explicitly.
pub(crate) fn explicit_drop_method_use(
    ctx: &DiagnosticsContext<'_, '_>,
    d: &hir::ExplicitDropMethodUse,
) -> Diagnostic {
    match d.expr_or_path {
        Either::Left(expr) => {
            let display_range = adjusted_display_range(ctx, expr, &|node| {
                Some(node.name_ref()?.syntax().text_range())
            });
            Diagnostic::new(
                DiagnosticCode::RustcHardError("E0040"),
                "explicit use of destructor method",
                display_range,
            )
            .stable()
            .with_main_node(expr.map(Into::into))
            .with_fixes(fix_method_call(ctx, expr))
        }
        Either::Right(path) => Diagnostic::new_with_syntax_node_ptr(
            ctx,
            DiagnosticCode::RustcHardError("E0040"),
            "explicit use of destructor method",
            path.map(Into::into),
        )
        .stable()
        .with_fixes(fix_path(ctx, path)),
    }
}

fn fix_method_call(
    ctx: &DiagnosticsContext<'_, '_>,
    mcall_ptr: InFile<AstPtr<ast::MethodCallExpr>>,
) -> Option<Vec<Assist>> {
    if mcall_ptr.file_id.is_macro() {
        // TODO: handle macro calls. Rough plan:
        // 1. upmap the range of the receiver and the range of the whole call
        // 2. delete everything outside the receiver and replace it with `drop(...)`, using range edits only.
        return None;
    }

    let db = ctx.db();

    let file_id = mcall_ptr.file_id;
    let mcall = mcall_ptr.to_node(db);
    let range = mcall.syntax().text_range();

    // `mcall` is `foo.drop()` -- extract the receiver, and wrap it in `drop()`
    // NOTE: it could theoretically be `(&mut foo).drop()` instead, in which case the fix
    // below would be incorrect, as it'd result in `drop((&mut foo))` instead of `drop(foo)`
    // -- but we don't bother to deal with that case.
    let recv = mcall.receiver()?;

    let mut builder = SourceChangeBuilder::new(file_id.original_file(db).file_id(db));
    let editor = builder.make_editor(mcall.syntax());
    let make = editor.make();
    let new_call =
        make.expr_call(make.expr_path(make.path_from_text("drop")), make.arg_list([recv]));
    builder.replace_ast(ast::Expr::MethodCallExpr(mcall), ast::Expr::CallExpr(new_call));
    let source_change = builder.finish();
    Some(vec![fix("use-drop-function", "Use `drop` function", source_change, range)])
}

fn fix_path(
    ctx: &DiagnosticsContext<'_, '_>,
    path_ptr: InFile<AstPtr<ast::Path>>,
) -> Option<Vec<Assist>> {
    let db = ctx.db();

    let file_id = path_ptr.file_id;
    let path = path_ptr.to_node(db);

    if let Some(call) =
        path.syntax().parent().and_then(|it| it.parent()).and_then(ast::CallExpr::cast)
    {
        if file_id.is_macro() {
            // TODO: make this work in macros? Might not be worth it, as this is a niche way to trigger this
            // already niche error
            return None;
        }

        // `call` is `Drop::drop(&mut foo)` -- extract the arg, and wrap it in `drop()`
        let arg_list = call.arg_list()?;
        let ref_recv = arg_list.args().exactly_one().ok()?;
        let ast::Expr::RefExpr(ref_recv) = ref_recv else {
            return None;
        };
        let recv = ref_recv.expr()?;

        let range = call.syntax().text_range();

        let mut builder = SourceChangeBuilder::new(file_id.original_file(db).file_id(db));
        let editor = builder.make_editor(call.syntax());
        let make = editor.make();
        let new_call =
            make.expr_call(make.expr_path(make.path_from_text("drop")), make.arg_list([recv]));
        builder.replace_ast(call, new_call);
        let source_change = builder.finish();
        Some(vec![fix("use-drop-function", "Use `drop` function", source_change, range)])
    } else {
        // `path` could be the `Foo::drop` in `let d = Foo::drop;`
        // -- replace the path with `drop`

        let range = InFile::new(file_id, path.syntax().text_range())
            .original_node_file_range_rooted_opt(db)?;

        let edit = TextEdit::replace(range.range, "drop".to_owned());
        let source_change = SourceChange::from_text_edit(range.file_id.file_id(db), edit);
        Some(vec![fix("use-drop-function", "Use `drop` function", source_change, range.range)])
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{
        check_diagnostics, check_diagnostics_with_disabled, check_fix, check_fix_with_disabled,
    };

    #[test]
    fn method_call_diagnostic() {
        check_diagnostics(
            r#"
//- minicore: drop
struct A;
impl Drop for A { fn drop(&mut self) {} }

fn main(mut a: A) {
    a.drop();
   // ^^^^ 💡 error: explicit use of destructor method
}
"#,
        );
    }

    #[test]
    fn method_call_fix() {
        check_fix(
            r#"
//- minicore: drop
struct A;
impl Drop for A { fn drop(&mut self) {} }

fn main(mut a: A) {
    a.drop$0();
}
"#,
            r#"
struct A;
impl Drop for A { fn drop(&mut self) {} }

fn main(mut a: A) {
    drop(a);
}
"#,
        );
    }

    #[test]
    fn qualified_call_1_diagnostic() {
        check_diagnostics(
            r#"
//- minicore: drop
struct A;
impl Drop for A { fn drop(&mut self) {} }

fn main(mut a: A) {
    A::drop(&mut a);
 // ^^^^^^^ 💡 error: explicit use of destructor method
}
"#,
        );
    }

    #[test]
    fn qualified_call_1_fix() {
        check_fix(
            r#"
//- minicore: drop
struct A;
impl Drop for A { fn drop(&mut self) {} }

fn main(mut a: A) {
    A::drop(&mut a$0);
}
"#,
            r#"
struct A;
impl Drop for A { fn drop(&mut self) {} }

fn main(mut a: A) {
    drop(a);
}
"#,
        )
    }

    #[test]
    fn qualified_call_2_diagnostic() {
        check_diagnostics(
            r#"
//- minicore: drop
struct A;
impl Drop for A { fn drop(&mut self) {} }

fn main(mut a: A) {
    Drop::drop(&mut a);
 // ^^^^^^^^^^ 💡 error: explicit use of destructor method
}
"#,
        );
    }

    #[test]
    fn qualified_call_2_fix() {
        check_fix(
            r#"
//- minicore: drop
struct A;
impl Drop for A { fn drop(&mut self) {} }

fn main(mut a: A) {
    Drop::drop(&mut a$0);
}
"#,
            r#"
struct A;
impl Drop for A { fn drop(&mut self) {} }

fn main(mut a: A) {
    drop(a);
}
"#,
        )
    }

    #[test]
    fn fully_qualified_call_diagnostic() {
        check_diagnostics(
            r#"
//- minicore: drop
struct A;
impl Drop for A { fn drop(&mut self) {} }

fn main(mut a: A) {
    <A as Drop>::drop(&mut a);
 // ^^^^^^^^^^^^^^^^^ 💡 error: explicit use of destructor method
}
"#,
        );
    }

    #[test]
    fn fully_qualified_call_fix() {
        check_fix(
            r#"
//- minicore: drop
struct A;
impl Drop for A { fn drop(&mut self) {} }

fn main(mut a: A) {
    <A as Drop>::drop(&mut a$0);
}
"#,
            r#"
struct A;
impl Drop for A { fn drop(&mut self) {} }

fn main(mut a: A) {
    drop(a);
}
"#,
        )
    }

    #[test]
    fn path_diagnostic() {
        check_diagnostics_with_disabled(
            r#"
//- minicore: drop
struct A;
impl Drop for A { fn drop(&mut self) {} }

fn main(mut a: A) {
    let d = A::drop;
         // ^^^^^^^ 💡 error: explicit use of destructor method
    d(&mut a);
}
"#,
            // Because of the error, the code isn't analyzed further (?), and so `d` is warned on as unused.
            // Arguably a bug in r-a (rustc doesn't emit a warning in this case)
            // FIXME: remove this once r-a no longer warns
            &["unused_variables"],
        );
    }

    #[test]
    // NOTE: Here, the fix is not completely correct, as it doesn't replace `d(&mut a)` with `d(a)`.
    // Oh well, rustc doesn't either
    fn path_fix() {
        check_fix_with_disabled(
            r#"
//- minicore: drop
struct A;
impl Drop for A { fn drop(&mut self) {} }

fn main(mut a: A) {
    let d = A::drop$0;
    d(&mut a);
}
"#,
            r#"
struct A;
impl Drop for A { fn drop(&mut self) {} }

fn main(mut a: A) {
    let d = drop;
    d(&mut a);
}
"#,
            // Because of the error, the code isn't analyzed further (?), and so `d` is warned on as unused.
            // Arguably a bug in r-a (rustc doesn't emit a warning in this case)
            // FIXME: remove this once r-a no longer warns
            &["unused_variables"],
        );
    }

    #[test]
    // NOTE: Here, the fix is not completely correct, as it doesn't replace `d(&mut a)` with `d(a)`.
    // Oh well, rustc doesn't either
    fn path_fix_in_macro() {
        check_fix(
            r#"
//- minicore: drop
struct A;
impl Drop for A { fn drop(&mut self) {} }

macro_rules! main {
    ($e:expr) => {
        fn main() { $e }
    }
}

main!{{
    let mut a = A;
    let d = A::drop$0;
    d(&mut a);
}};
"#,
            r#"
struct A;
impl Drop for A { fn drop(&mut self) {} }

macro_rules! main {
    ($e:expr) => {
        fn main() { $e }
    }
}

main!{{
    let mut a = A;
    let d = drop;
    d(&mut a);
}};
"#,
        );
    }

    #[test]
    fn std_mem_drop() {
        check_diagnostics(
            r#"
//- minicore: drop
struct A;
impl Drop for A { fn drop(&mut self) {} }

fn main(a: A) {
    drop(a);
}
"#,
        );
    }

    #[test]
    fn inherent_drop_method() {
        check_diagnostics(
            r#"
struct A;
impl A { fn drop(&mut self) {} }

fn main(mut a: A) {
    a.drop();
}
"#,
        );
    }

    #[test]
    fn custom_trait_drop_method() {
        check_diagnostics(
            r#"
struct A;
trait MyDrop { fn drop(&mut self); }
impl MyDrop for A { fn drop(&mut self) {} }

fn main(mut a: A) {
    a.drop();
}
"#,
        );
    }
}
