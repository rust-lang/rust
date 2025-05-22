use either::Either;
use hir::InFile;
use ide_db::FileRange;
use syntax::{
    AstNode, AstPtr,
    ast::{self, HasArgList},
};

use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext, adjusted_display_range};

// Diagnostic: mismatched-tuple-struct-pat-arg-count
//
// This diagnostic is triggered if a function is invoked with an incorrect amount of arguments.
pub(crate) fn mismatched_tuple_struct_pat_arg_count(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::MismatchedTupleStructPatArgCount,
) -> Diagnostic {
    let s = if d.found == 1 { "" } else { "s" };
    let s2 = if d.expected == 1 { "" } else { "s" };
    let message = format!(
        "this pattern has {} field{s}, but the corresponding tuple struct has {} field{s2}",
        d.found, d.expected
    );
    Diagnostic::new(
        DiagnosticCode::RustcHardError("E0023"),
        message,
        invalid_args_range(ctx, d.expr_or_pat, d.expected, d.found),
    )
    .stable()
}

// Diagnostic: mismatched-arg-count
//
// This diagnostic is triggered if a function is invoked with an incorrect amount of arguments.
pub(crate) fn mismatched_arg_count(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::MismatchedArgCount,
) -> Diagnostic {
    let s = if d.expected == 1 { "" } else { "s" };
    let message = format!("expected {} argument{s}, found {}", d.expected, d.found);
    Diagnostic::new(
        DiagnosticCode::RustcHardError("E0107"),
        message,
        invalid_args_range(ctx, d.call_expr, d.expected, d.found),
    )
    .stable()
}

fn invalid_args_range(
    ctx: &DiagnosticsContext<'_>,
    source: InFile<AstPtr<Either<ast::Expr, ast::Pat>>>,
    expected: usize,
    found: usize,
) -> FileRange {
    adjusted_display_range(ctx, source, &|expr| {
        let (text_range, r_paren_token, expected_arg) = match expr {
            Either::Left(ast::Expr::CallExpr(call)) => {
                let arg_list = call.arg_list()?;
                (
                    arg_list.syntax().text_range(),
                    arg_list.r_paren_token(),
                    arg_list.args().nth(expected).map(|it| it.syntax().text_range()),
                )
            }
            Either::Left(ast::Expr::MethodCallExpr(call)) => {
                let arg_list = call.arg_list()?;
                (
                    arg_list.syntax().text_range(),
                    arg_list.r_paren_token(),
                    arg_list.args().nth(expected).map(|it| it.syntax().text_range()),
                )
            }
            Either::Right(ast::Pat::TupleStructPat(pat)) => {
                let r_paren = pat.r_paren_token()?;
                let l_paren = pat.l_paren_token()?;
                (
                    l_paren.text_range().cover(r_paren.text_range()),
                    Some(r_paren),
                    pat.fields().nth(expected).map(|it| it.syntax().text_range()),
                )
            }
            _ => return None,
        };
        if found < expected {
            if found == 0 {
                return Some(text_range);
            }
            if let Some(r_paren) = r_paren_token {
                return Some(r_paren.text_range());
            }
        }
        if expected < found {
            if expected == 0 {
                return Some(text_range);
            }
            let zip = expected_arg.zip(r_paren_token);
            if let Some((arg, r_paren)) = zip {
                return Some(arg.cover(r_paren.text_range()));
            }
        }

        None
    })
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
fn one(_arg: u8) {}
fn f() { one(); }
          //^^ error: expected 1 argument, found 0
"#,
        );

        check_diagnostics(
            r#"
fn one(_arg: u8) {}
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
impl S { fn method(&self, _arg: u8) {} }

            fn f() {
                S.method();
            }         //^^ error: expected 1 argument, found 0
            "#,
        );

        check_diagnostics(
            r#"
struct S;
impl S { fn method(&self, _arg: u8) {} }

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
trait Foo { fn method(&self, _arg: usize) {} }

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
                  //^^ error: expected 1 argument, found 2
        Foo::Bar();
              //^^ error: expected 1 argument, found 0
    }
}
        "#,
        );
    }

    #[test]
    fn rest_pat_in_macro_expansion() {
        check_diagnostics(
            r#"
// issue #17292
#![allow(dead_code)]

macro_rules! replace_with_2_dots {
    ( $( $input:tt )* ) => {
        ..
    };
}

macro_rules! enum_str {
    (
        $(
            $variant:ident (
                $( $tfield:ty ),*
            )
        )
        ,
        *
    ) => {
        enum Foo {
            $(
                $variant ( $( $tfield ),* ),
            )*
        }

        impl Foo {
            fn variant_name_as_str(&self) -> &str {
                match self {
                    $(
                        Self::$variant ( replace_with_2_dots!( $( $tfield ),* ) )
                          => "",
                    )*
                }
            }
        }
    };
}

enum_str! {
    TupleVariant1(i32),
    TupleVariant2(),
    TupleVariant3(i8,u8,i128)
}
"#,
        );

        check_diagnostics(
            r#"
#![allow(dead_code)]
macro_rules! two_dots1 {
    () => { .. };
}

macro_rules! two_dots2 {
    () => { two_dots1!() };
}

fn test() {
    let (_, _, two_dots1!()) = ((), 42);
    let (_, two_dots2!(), _) = (1, true, 2, false, (), (), 3);
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
               //^^ error: expected 1 argument, found 2
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
        //^^^ error: expected 1 argument, found 2
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
    fn method2(#[cfg(NEVER)] self, _arg: u8) {}
    fn method3(self, #[cfg(NEVER)] _arg: u8) {}
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

    #[test]
    fn legacy_const_generics() {
        check_diagnostics(
            r#"
#[rustc_legacy_const_generics(1, 3)]
fn mixed<const N1: &'static str, const N2: bool>(
    _a: u8,
    _b: i8,
) {}

fn f() {
    mixed(0, "", -1, true);
    mixed::<"", true>(0, -1);
}

#[rustc_legacy_const_generics(1, 3)]
fn b<const N1: u8, const N2: u8>(
    _a: u8,
    _b: u8,
) {}

fn g() {
    b(0, 1, 2, 3);
    b::<1, 3>(0, 2);

    b(0, 1, 2);
           //^ error: expected 4 arguments, found 3
}
            "#,
        )
    }

    #[test]
    fn tuple_struct_pat() {
        check_diagnostics(
            r#"
struct S(u32, u32);
fn f(
    S(a, b, c): S,
         // ^^ error: this pattern has 3 fields, but the corresponding tuple struct has 2 fields
    S(): S,
  // ^^ error: this pattern has 0 fields, but the corresponding tuple struct has 2 fields
    S(e, f, .., g, d): S
  //        ^^^^^^^^^ error: this pattern has 4 fields, but the corresponding tuple struct has 2 fields
) { _ = (a, b, c, d, e, f, g); }
"#,
        )
    }

    #[test]
    fn no_type_mismatches_when_arg_count_mismatch() {
        check_diagnostics(
            r#"
fn foo((): (), (): ()) {
    foo(1, 2, 3);
           // ^^ error: expected 2 arguments, found 3
    foo(1);
      // ^ error: expected 2 arguments, found 1
}
"#,
        );
    }
}
