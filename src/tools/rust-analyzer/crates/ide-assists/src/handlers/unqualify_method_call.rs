use hir::AsAssocItem;
use syntax::ast::{self, AstNode, HasArgList, prec::ExprPrecedence, syntax_factory::SyntaxFactory};

use crate::{AssistContext, AssistId, Assists};

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
    let first_arg = args.args().next()?;

    let qualifier = path.qualifier()?;
    let method_name = path.segment()?.name_ref()?;

    let scope = ctx.sema.scope(path.syntax())?;
    let res = ctx.sema.resolve_path(&path)?;
    let hir::PathResolution::Def(hir::ModuleDef::Function(fun)) = res else { return None };
    if !fun.has_self_param(ctx.sema.db) {
        return None;
    }

    acc.add(
        AssistId::refactor_rewrite("unqualify_method_call"),
        "Unqualify method call",
        call.syntax().text_range(),
        |builder| {
            let make = SyntaxFactory::with_mappings();
            let mut editor = builder.make_editor(call.syntax());

            let new_arg_list = make.arg_list(args.args().skip(1));
            let receiver = if first_arg.precedence().needs_parentheses_in(ExprPrecedence::Postfix) {
                ast::Expr::from(make.expr_paren(first_arg.clone()))
            } else {
                first_arg.clone()
            };
            let method_call = make.expr_method_call(receiver, method_name, new_arg_list);

            editor.replace(call.syntax(), method_call.syntax());

            if let Some(fun) = fun.as_assoc_item(ctx.db())
                && let Some(trait_) = fun.container_or_implemented_trait(ctx.db())
                && !scope.can_use_trait_methods(trait_)
            {
                add_import(qualifier, ctx, &make, &mut editor);
            }

            editor.add_mappings(make.finish_with_mappings());
            builder.add_file_edits(ctx.vfs_file_id(), editor);
        },
    )
}

fn add_import(
    qualifier: ast::Path,
    ctx: &AssistContext<'_>,
    make: &SyntaxFactory,
    editor: &mut syntax::syntax_editor::SyntaxEditor,
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
            ide_db::imports::insert_use::insert_use_with_editor(
                &scope,
                import,
                &ctx.config.insert_use,
                editor,
                make,
            );
        }
    }
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

    #[test]
    fn inherent_method() {
        check_assist(
            unqualify_method_call,
            r#"
mod foo {
    pub struct Bar;
    impl Bar {
        pub fn bar(self) {}
    }
}

fn baz() {
    foo::Bar::b$0ar(foo::Bar);
}
        "#,
            r#"
mod foo {
    pub struct Bar;
    impl Bar {
        pub fn bar(self) {}
    }
}

fn baz() {
    foo::Bar.bar();
}
        "#,
        );
    }

    #[test]
    fn trait_method_in_impl() {
        check_assist(
            unqualify_method_call,
            r#"
mod foo {
    pub trait Bar {
        pub fn bar(self) {}
    }
}

struct Baz;
impl foo::Bar for Baz {
    fn bar(self) {
        foo::Bar::b$0ar(Baz);
    }
}
        "#,
            r#"
mod foo {
    pub trait Bar {
        pub fn bar(self) {}
    }
}

struct Baz;
impl foo::Bar for Baz {
    fn bar(self) {
        Baz.bar();
    }
}
        "#,
        );
    }

    #[test]
    fn trait_method_already_imported() {
        check_assist(
            unqualify_method_call,
            r#"
mod foo {
    pub struct Foo;
    pub trait Bar {
        pub fn bar(self) {}
    }
    impl Bar for Foo {
        pub fn bar(self) {}
    }
}

use foo::Bar;

fn baz() {
    foo::Bar::b$0ar(foo::Foo);
}
        "#,
            r#"
mod foo {
    pub struct Foo;
    pub trait Bar {
        pub fn bar(self) {}
    }
    impl Bar for Foo {
        pub fn bar(self) {}
    }
}

use foo::Bar;

fn baz() {
    foo::Foo.bar();
}
        "#,
        );
    }
}
