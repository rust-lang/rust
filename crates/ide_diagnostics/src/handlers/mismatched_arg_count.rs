use ide_db::base_db::{FileRange, SourceDatabase};
use syntax::{
    algo::find_node_at_range,
    ast::{self, HasArgList},
    AstNode,
};

use crate::{Diagnostic, DiagnosticsContext};

// Diagnostic: mismatched-arg-count
//
// This diagnostic is triggered if a function is invoked with an incorrect amount of arguments.
pub(crate) fn mismatched_arg_count(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::MismatchedArgCount,
) -> Diagnostic {
    let s = if d.expected == 1 { "" } else { "s" };
    let message = format!("expected {} argument{}, found {}", d.expected, s, d.found);
    Diagnostic::new(
        "mismatched-arg-count",
        message,
        invalid_args_range(ctx, d).unwrap_or_else(|it| it).range,
    )
}

fn invalid_args_range(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::MismatchedArgCount,
) -> Result<FileRange, FileRange> {
    let full_range = ctx.sema.diagnostics_display_range(d.call_expr.clone().map(|it| it.into()));

    let source_file = ctx.sema.db.parse(full_range.file_id);
    let expr = find_node_at_range::<ast::Expr>(&source_file.syntax_node(), full_range.range)
        .filter(|it| it.syntax().text_range() == full_range.range);
    let arg_list = match expr {
        Some(ast::Expr::CallExpr(call)) => call.arg_list(),
        Some(ast::Expr::MethodCallExpr(call)) => call.arg_list(),
        _ => None,
    };
    let arg_list = match arg_list {
        Some(it) => it,
        None => return Err(full_range),
    };
    let arg_list_range =
        FileRange { file_id: full_range.file_id, range: arg_list.syntax().text_range() };
    if d.found < d.expected {
        if d.found == 0 {
            return Ok(arg_list_range);
        }
        if let Some(r_paren) = arg_list.r_paren_token() {
            return Ok(FileRange { file_id: full_range.file_id, range: r_paren.text_range() });
        }
    }
    if d.expected < d.found {
        return Ok(arg_list_range);
    }

    Err(full_range)
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn simple_free_fn_zero() {
        check_diagnostics(
            r#"
fn zero() {}
fn f() { zero(1); }
           //^^^ error: expected 0 arguments, found 1
"#,
        );

        check_diagnostics(
            r#"
fn zero() {}
fn f() { zero(); }
"#,
        );
    }

    #[test]
    fn simple_free_fn_one() {
        check_diagnostics(
            r#"
fn one(arg: u8) {}
fn f() { one(); }
          //^^ error: expected 1 argument, found 0
"#,
        );

        check_diagnostics(
            r#"
fn one(arg: u8) {}
fn f() { one(1); }
"#,
        );
    }

    #[test]
    fn method_as_fn() {
        check_diagnostics(
            r#"
struct S;
impl S { fn method(&self) {} }

fn f() {
    S::method();
}          //^^ error: expected 1 argument, found 0
"#,
        );

        check_diagnostics(
            r#"
struct S;
impl S { fn method(&self) {} }

fn f() {
    S::method(&S);
    S.method();
}
"#,
        );
    }

    #[test]
    fn method_with_arg() {
        check_diagnostics(
            r#"
struct S;
impl S { fn method(&self, arg: u8) {} }

            fn f() {
                S.method();
            }         //^^ error: expected 1 argument, found 0
            "#,
        );

        check_diagnostics(
            r#"
struct S;
impl S { fn method(&self, arg: u8) {} }

fn f() {
    S::method(&S, 0);
    S.method(1);
}
"#,
        );
    }

    #[test]
    fn method_unknown_receiver() {
        // note: this is incorrect code, so there might be errors on this in the
        // future, but we shouldn't emit an argument count diagnostic here
        check_diagnostics(
            r#"
trait Foo { fn method(&self, arg: usize) {} }

fn f() {
    let x;
    x.method();
}
"#,
        );
    }

    #[test]
    fn tuple_struct() {
        check_diagnostics(
            r#"
struct Tup(u8, u16);
fn f() {
    Tup(0);
}      //^ error: expected 2 arguments, found 1
"#,
        )
    }

    #[test]
    fn enum_variant() {
        check_diagnostics(
            r#"
enum En { Variant(u8, u16), }
fn f() {
    En::Variant(0);
}              //^ error: expected 2 arguments, found 1
"#,
        )
    }

    #[test]
    fn enum_variant_type_macro() {
        check_diagnostics(
            r#"
macro_rules! Type {
    () => { u32 };
}
enum Foo {
    Bar(Type![])
}
impl Foo {
    fn new() {
        Foo::Bar(0);
        Foo::Bar(0, 1);
              //^^^^^^ error: expected 1 argument, found 2
        Foo::Bar();
              //^^ error: expected 1 argument, found 0
    }
}
        "#,
        );
    }

    #[test]
    fn varargs() {
        check_diagnostics(
            r#"
extern "C" {
    fn fixed(fixed: u8);
    fn varargs(fixed: u8, ...);
    fn varargs2(...);
}

fn f() {
    unsafe {
        fixed(0);
        fixed(0, 1);
           //^^^^^^ error: expected 1 argument, found 2
        varargs(0);
        varargs(0, 1);
        varargs2();
        varargs2(0);
        varargs2(0, 1);
    }
}
        "#,
        )
    }

    #[test]
    fn arg_count_lambda() {
        check_diagnostics(
            r#"
fn main() {
    let f = |()| ();
    f();
   //^^ error: expected 1 argument, found 0
    f(());
    f((), ());
   //^^^^^^^^ error: expected 1 argument, found 2
}
"#,
        )
    }

    #[test]
    fn cfgd_out_call_arguments() {
        check_diagnostics(
            r#"
struct C(#[cfg(FALSE)] ());
impl C {
    fn new() -> Self {
        Self(
            #[cfg(FALSE)]
            (),
        )
    }

    fn method(&self) {}
}

fn main() {
    C::new().method(#[cfg(FALSE)] 0);
}
            "#,
        );
    }

    #[test]
    fn cfgd_out_fn_params() {
        check_diagnostics(
            r#"
fn foo(#[cfg(NEVER)] x: ()) {}

struct S;

impl S {
    fn method(#[cfg(NEVER)] self) {}
    fn method2(#[cfg(NEVER)] self, arg: u8) {}
    fn method3(self, #[cfg(NEVER)] arg: u8) {}
}

extern "C" {
    fn fixed(fixed: u8, #[cfg(NEVER)] ...);
    fn varargs(#[cfg(not(NEVER))] ...);
}

fn main() {
    foo();
    S::method();
    S::method2(0);
    S::method3(S);
    S.method3();
    unsafe {
        fixed(0);
        varargs(1, 2, 3);
    }
}
            "#,
        )
    }
}
