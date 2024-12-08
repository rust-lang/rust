use ide_db::imports::insert_use::ImportScope;
use syntax::{
    ast::{self, make, AstNode, HasArgList},
    TextRange,
};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: unqualify_method_call
//
// Transforms universal function call syntax into a method call.
//
// ```
// fn main() {
//     std::ops::Add::add$0(1, 2);
// }
// # mod std { pub mod ops { pub trait Add { fn add(self, _: Self) {} } impl Add for i32 {} } }
// ```
// ->
// ```
// use std::ops::Add;
//
// fn main() {
//     1.add(2);
// }
// # mod std { pub mod ops { pub trait Add { fn add(self, _: Self) {} } impl Add for i32 {} } }
// ```
pub(crate) fn unqualify_method_call(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let call = ctx.find_node_at_offset::<ast::CallExpr>()?;
    let ast::Expr::PathExpr(path_expr) = call.expr()? else { return None };
    let path = path_expr.path()?;

    let cursor_in_range = path.syntax().text_range().contains_range(ctx.selection_trimmed());
    if !cursor_in_range {
        return None;
    }

    let args = call.arg_list()?;
    let l_paren = args.l_paren_token()?;
    let mut args_iter = args.args();
    let first_arg = args_iter.next()?;
    let second_arg = args_iter.next();

    let qualifier = path.qualifier()?;
    let method_name = path.segment()?.name_ref()?;

    let res = ctx.sema.resolve_path(&path)?;
    let hir::PathResolution::Def(hir::ModuleDef::Function(fun)) = res else { return None };
    if !fun.has_self_param(ctx.sema.db) {
        return None;
    }

    // `core::ops::Add::add(` -> ``
    let delete_path =
        TextRange::new(path.syntax().text_range().start(), l_paren.text_range().end());

    // Parens around `expr` if needed
    let parens = needs_parens_as_receiver(&first_arg).then(|| {
        let range = first_arg.syntax().text_range();
        (range.start(), range.end())
    });

    // `, ` -> `.add(`
    let replace_comma = TextRange::new(
        first_arg.syntax().text_range().end(),
        second_arg
            .map(|a| a.syntax().text_range().start())
            .unwrap_or_else(|| first_arg.syntax().text_range().end()),
    );

    acc.add(
        AssistId("unqualify_method_call", AssistKind::RefactorRewrite),
        "Unqualify method call",
        call.syntax().text_range(),
        |edit| {
            edit.delete(delete_path);
            if let Some((open, close)) = parens {
                edit.insert(open, "(");
                edit.insert(close, ")");
            }
            edit.replace(replace_comma, format!(".{method_name}("));
            add_import(qualifier, ctx, edit);
        },
    )
}

fn add_import(
    qualifier: ast::Path,
    ctx: &AssistContext<'_>,
    edit: &mut ide_db::source_change::SourceChangeBuilder,
) {
    if let Some(path_segment) = qualifier.segment() {
        // for `<i32 as std::ops::Add>`
        let path_type = path_segment.qualifying_trait();
        let import = match path_type {
            Some(it) => {
                if let Some(path) = it.path() {
                    path
                } else {
                    return;
                }
            }
            None => qualifier,
        };

        // in case for `<_>`
        if import.coloncolon_token().is_none() {
            return;
        }

        let scope = ide_db::imports::insert_use::ImportScope::find_insert_use_container(
            import.syntax(),
            &ctx.sema,
        );

        if let Some(scope) = scope {
            let scope = match scope {
                ImportScope::File(it) => ImportScope::File(edit.make_mut(it)),
                ImportScope::Module(it) => ImportScope::Module(edit.make_mut(it)),
                ImportScope::Block(it) => ImportScope::Block(edit.make_mut(it)),
            };
            ide_db::imports::insert_use::insert_use(&scope, import, &ctx.config.insert_use);
        }
    }
}

fn needs_parens_as_receiver(expr: &ast::Expr) -> bool {
    // Make `(expr).dummy()`
    let dummy_call = make::expr_method_call(
        make::expr_paren(expr.clone()),
        make::name_ref("dummy"),
        make::arg_list([]),
    );

    // Get the `expr` clone with the right parent back
    // (unreachable!s are fine since we've just constructed the expression)
    let ast::Expr::MethodCallExpr(call) = &dummy_call else { unreachable!() };
    let Some(receiver) = call.receiver() else { unreachable!() };
    let ast::Expr::ParenExpr(parens) = receiver else { unreachable!() };
    let Some(expr) = parens.expr() else { unreachable!() };

    expr.needs_parens_in(dummy_call.syntax().clone())
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn unqualify_method_call_simple() {
        check_assist(
            unqualify_method_call,
            r#"
struct S;
impl S { fn f(self, S: S) {} }
fn f() { S::$0f(S, S); }"#,
            r#"
struct S;
impl S { fn f(self, S: S) {} }
fn f() { S.f(S); }"#,
        );
    }

    #[test]
    fn unqualify_method_call_trait() {
        check_assist(
            unqualify_method_call,
            r#"
//- minicore: add
fn f() { <u32 as core::ops::Add>::$0add(2, 2); }"#,
            r#"
use core::ops::Add;

fn f() { 2.add(2); }"#,
        );

        check_assist(
            unqualify_method_call,
            r#"
//- minicore: add
fn f() { core::ops::Add::$0add(2, 2); }"#,
            r#"
use core::ops::Add;

fn f() { 2.add(2); }"#,
        );

        check_assist(
            unqualify_method_call,
            r#"
//- minicore: add
use core::ops::Add;
fn f() { <_>::$0add(2, 2); }"#,
            r#"
use core::ops::Add;
fn f() { 2.add(2); }"#,
        );
    }

    #[test]
    fn unqualify_method_call_single_arg() {
        check_assist(
            unqualify_method_call,
            r#"
        struct S;
        impl S { fn f(self) {} }
        fn f() { S::$0f(S); }"#,
            r#"
        struct S;
        impl S { fn f(self) {} }
        fn f() { S.f(); }"#,
        );
    }

    #[test]
    fn unqualify_method_call_parens() {
        check_assist(
            unqualify_method_call,
            r#"
//- minicore: deref
struct S;
impl core::ops::Deref for S {
    type Target = S;
    fn deref(&self) -> &S { self }
}
fn f() { core::ops::Deref::$0deref(&S); }"#,
            r#"
use core::ops::Deref;

struct S;
impl core::ops::Deref for S {
    type Target = S;
    fn deref(&self) -> &S { self }
}
fn f() { (&S).deref(); }"#,
        );
    }

    #[test]
    fn unqualify_method_call_doesnt_apply_with_cursor_not_on_path() {
        check_assist_not_applicable(
            unqualify_method_call,
            r#"
//- minicore: add
fn f() { core::ops::Add::add(2,$0 2); }"#,
        );
    }

    #[test]
    fn unqualify_method_call_doesnt_apply_with_no_self() {
        check_assist_not_applicable(
            unqualify_method_call,
            r#"
struct S;
impl S { fn assoc(S: S, S: S) {} }
fn f() { S::assoc$0(S, S); }"#,
        );
    }
}
