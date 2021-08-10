use ast::make;
use hir::{HasSource, PathResolution, TypeInfo};
use ide_db::{defs::Definition, path_transform::PathTransform, search::FileReference};
use itertools::izip;
use syntax::{
    ast::{self, edit::AstNodeEdit, ArgListOwner},
    ted, AstNode,
};

use crate::{
    assist_context::{AssistContext, Assists},
    AssistId, AssistKind,
};

// Assist: inline_call
//
// Inlines a function or method body creating a `let` statement per parameter unless the parameter
// can be inlined. The parameter will be inlined either if it the supplied argument is a simple local
// or if the parameter is only accessed inside the function body once.
//
// ```
// # //- minicore: option
// fn foo(name: Option<&str>) {
//     let name = name.unwrap$0();
// }
// ```
// ->
// ```
// fn foo(name: Option<&str>) {
//     let name = match name {
//             Some(val) => val,
//             None => panic!("called `Option::unwrap()` on a `None` value"),
//         };
// }
// ```
pub(crate) fn inline_call(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let (label, function, arguments, generic_arg_list, expr) =
        if let Some(path_expr) = ctx.find_node_at_offset::<ast::PathExpr>() {
            let call = path_expr.syntax().parent().and_then(ast::CallExpr::cast)?;
            let path = path_expr.path()?;

            let function = match ctx.sema.resolve_path(&path)? {
                PathResolution::Def(hir::ModuleDef::Function(f))
                | PathResolution::AssocItem(hir::AssocItem::Function(f)) => f,
                _ => return None,
            };
            (
                format!("Inline `{}`", path),
                function,
                call.arg_list()?.args().collect(),
                path.segment().and_then(|it| it.generic_arg_list()),
                ast::Expr::CallExpr(call),
            )
        } else {
            let name_ref: ast::NameRef = ctx.find_node_at_offset()?;
            let call = name_ref.syntax().parent().and_then(ast::MethodCallExpr::cast)?;
            let receiver = call.receiver()?;
            let function = ctx.sema.resolve_method_call(&call)?;
            let mut arguments = vec![receiver];
            arguments.extend(call.arg_list()?.args());
            (
                format!("Inline `{}`", name_ref),
                function,
                arguments,
                call.generic_arg_list(),
                ast::Expr::MethodCallExpr(call),
            )
        };

    inline_(acc, ctx, label, function, arguments, expr, generic_arg_list)
}

pub(crate) fn inline_(
    acc: &mut Assists,
    ctx: &AssistContext,
    label: String,
    function: hir::Function,
    arg_list: Vec<ast::Expr>,
    expr: ast::Expr,
    generic_arg_list: Option<ast::GenericArgList>,
) -> Option<()> {
    let hir::InFile { value: function_source, file_id } = function.source(ctx.db())?;
    let param_list = function_source.param_list()?;
    let mut assoc_fn_params = function.assoc_fn_params(ctx.sema.db).into_iter();

    let mut params = Vec::new();
    if let Some(self_param) = param_list.self_param() {
        // FIXME this should depend on the receiver as well as the self_param
        params.push((
            make::ident_pat(
                self_param.amp_token().is_some(),
                self_param.mut_token().is_some(),
                make::name("this"),
            )
            .into(),
            None,
            assoc_fn_params.next()?,
        ));
    }
    for param in param_list.params() {
        params.push((param.pat()?, param.ty(), assoc_fn_params.next()?));
    }

    if arg_list.len() != params.len() {
        // Can't inline the function because they've passed the wrong number of
        // arguments to this function
        cov_mark::hit!(inline_call_incorrect_number_of_arguments);
        return None;
    }

    let fn_body = function_source.body()?;

    acc.add(
        AssistId("inline_call", AssistKind::RefactorInline),
        label,
        expr.syntax().text_range(),
        |builder| {
            let body = fn_body.clone_for_update();

            let file_id = file_id.original_file(ctx.sema.db);
            let usages_for_locals = |local| {
                Definition::Local(local)
                    .usages(&ctx.sema)
                    .all()
                    .references
                    .remove(&file_id)
                    .unwrap_or_default()
                    .into_iter()
            };
            // Contains the nodes of usages of parameters.
            // If the inner Vec for a parameter is empty it either means there are no usages or that the parameter
            // has a pattern that does not allow inlining
            let param_use_nodes: Vec<Vec<_>> = params
                .iter()
                .map(|(pat, _, param)| {
                    if !matches!(pat, ast::Pat::IdentPat(pat) if pat.is_simple_ident()) {
                        return Vec::new();
                    }
                    usages_for_locals(param.as_local(ctx.sema.db))
                        .map(|FileReference { name, range, .. }| match name {
                            ast::NameLike::NameRef(_) => body
                                .syntax()
                                .covering_element(range)
                                .ancestors()
                                .nth(3)
                                .and_then(ast::PathExpr::cast),
                            _ => None,
                        })
                        .collect::<Option<Vec<_>>>()
                        .unwrap_or_default()
                })
                .collect();

            // Rewrite `self` to `this`
            if param_list.self_param().is_some() {
                let this = || make::name_ref("this").syntax().clone_for_update();
                usages_for_locals(params[0].2.as_local(ctx.sema.db))
                    .flat_map(|FileReference { name, range, .. }| match name {
                        ast::NameLike::NameRef(_) => Some(body.syntax().covering_element(range)),
                        _ => None,
                    })
                    .for_each(|it| {
                        ted::replace(it, &this());
                    })
            }

            // Inline parameter expressions or generate `let` statements depending on whether inlining works or not.
            for ((pat, param_ty, _), usages, expr) in izip!(params, param_use_nodes, arg_list).rev()
            {
                let expr_is_name_ref = matches!(&expr,
                    ast::Expr::PathExpr(expr)
                        if expr.path().and_then(|path| path.as_single_name_ref()).is_some()
                );
                match &*usages {
                    // inline single use closure arguments
                    [usage]
                        if matches!(expr, ast::Expr::ClosureExpr(_))
                            && usage.syntax().parent().and_then(ast::Expr::cast).is_some() =>
                    {
                        cov_mark::hit!(inline_call_inline_closure);
                        let expr = make::expr_paren(expr);
                        ted::replace(usage.syntax(), expr.syntax().clone_for_update());
                    }
                    // inline single use literals
                    [usage] if matches!(expr, ast::Expr::Literal(_)) => {
                        cov_mark::hit!(inline_call_inline_literal);
                        ted::replace(usage.syntax(), expr.syntax().clone_for_update());
                    }
                    // inline direct local arguments
                    [_, ..] if expr_is_name_ref => {
                        cov_mark::hit!(inline_call_inline_locals);
                        usages.into_iter().for_each(|usage| {
                            ted::replace(usage.syntax(), &expr.syntax().clone_for_update());
                        });
                    }
                    // cant inline, emit a let statement
                    _ => {
                        let ty = ctx
                            .sema
                            .type_of_expr(&expr)
                            .filter(TypeInfo::has_adjustment)
                            .and_then(|_| param_ty);
                        body.push_front(
                            make::let_stmt(pat, ty, Some(expr)).clone_for_update().into(),
                        )
                    }
                }
            }
            if let Some(generic_arg_list) = generic_arg_list {
                PathTransform::function_call(
                    &ctx.sema.scope(expr.syntax()),
                    &ctx.sema.scope(fn_body.syntax()),
                    function,
                    generic_arg_list,
                )
                .apply(body.syntax());
            }

            let original_indentation = expr.indent_level();
            let replacement = body.reset_indent().indent(original_indentation);

            let replacement = match replacement.tail_expr() {
                Some(expr) if replacement.statements().next().is_none() => expr,
                _ => ast::Expr::BlockExpr(replacement),
            };
            builder.replace_ast(expr, replacement);
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn no_args_or_return_value_gets_inlined_without_block() {
        check_assist(
            inline_call,
            r#"
fn foo() { println!("Hello, World!"); }
fn main() {
    fo$0o();
}
"#,
            r#"
fn foo() { println!("Hello, World!"); }
fn main() {
    { println!("Hello, World!"); };
}
"#,
        );
    }

    #[test]
    fn not_applicable_when_incorrect_number_of_parameters_are_provided() {
        cov_mark::check!(inline_call_incorrect_number_of_arguments);
        check_assist_not_applicable(
            inline_call,
            r#"
fn add(a: u32, b: u32) -> u32 { a + b }
fn main() { let x = add$0(42); }
"#,
        );
    }

    #[test]
    fn args_with_side_effects() {
        check_assist(
            inline_call,
            r#"
fn foo(name: String) {
    println!("Hello, {}!", name);
}
fn main() {
    foo$0(String::from("Michael"));
}
"#,
            r#"
fn foo(name: String) {
    println!("Hello, {}!", name);
}
fn main() {
    {
        let name = String::from("Michael");
        println!("Hello, {}!", name);
    };
}
"#,
        );
    }

    #[test]
    fn function_with_multiple_statements() {
        check_assist(
            inline_call,
            r#"
fn foo(a: u32, b: u32) -> u32 {
    let x = a + b;
    let y = x - b;
    x * y
}

fn main() {
    let x = foo$0(1, 2);
}
"#,
            r#"
fn foo(a: u32, b: u32) -> u32 {
    let x = a + b;
    let y = x - b;
    x * y
}

fn main() {
    let x = {
        let b = 2;
        let x = 1 + b;
        let y = x - b;
        x * y
    };
}
"#,
        );
    }

    #[test]
    fn function_with_self_param() {
        check_assist(
            inline_call,
            r#"
struct Foo(u32);

impl Foo {
    fn add(self, a: u32) -> Self {
        Foo(self.0 + a)
    }
}

fn main() {
    let x = Foo::add$0(Foo(3), 2);
}
"#,
            r#"
struct Foo(u32);

impl Foo {
    fn add(self, a: u32) -> Self {
        Foo(self.0 + a)
    }
}

fn main() {
    let x = {
        let this = Foo(3);
        Foo(this.0 + 2)
    };
}
"#,
        );
    }

    #[test]
    fn method_by_val() {
        check_assist(
            inline_call,
            r#"
struct Foo(u32);

impl Foo {
    fn add(self, a: u32) -> Self {
        Foo(self.0 + a)
    }
}

fn main() {
    let x = Foo(3).add$0(2);
}
"#,
            r#"
struct Foo(u32);

impl Foo {
    fn add(self, a: u32) -> Self {
        Foo(self.0 + a)
    }
}

fn main() {
    let x = {
        let this = Foo(3);
        Foo(this.0 + 2)
    };
}
"#,
        );
    }

    #[test]
    fn method_by_ref() {
        check_assist(
            inline_call,
            r#"
struct Foo(u32);

impl Foo {
    fn add(&self, a: u32) -> Self {
        Foo(self.0 + a)
    }
}

fn main() {
    let x = Foo(3).add$0(2);
}
"#,
            r#"
struct Foo(u32);

impl Foo {
    fn add(&self, a: u32) -> Self {
        Foo(self.0 + a)
    }
}

fn main() {
    let x = {
        let ref this = Foo(3);
        Foo(this.0 + 2)
    };
}
"#,
        );
    }

    #[test]
    fn method_by_ref_mut() {
        check_assist(
            inline_call,
            r#"
struct Foo(u32);

impl Foo {
    fn clear(&mut self) {
        self.0 = 0;
    }
}

fn main() {
    let mut foo = Foo(3);
    foo.clear$0();
}
"#,
            r#"
struct Foo(u32);

impl Foo {
    fn clear(&mut self) {
        self.0 = 0;
    }
}

fn main() {
    let mut foo = Foo(3);
    {
        let ref mut this = foo;
        this.0 = 0;
    };
}
"#,
        );
    }

    #[test]
    fn function_multi_use_expr_in_param() {
        check_assist(
            inline_call,
            r#"
fn square(x: u32) -> u32 {
    x * x
}
fn main() {
    let x = 51;
    let y = square$0(10 + x);
}
"#,
            r#"
fn square(x: u32) -> u32 {
    x * x
}
fn main() {
    let x = 51;
    let y = {
        let x = 10 + x;
        x * x
    };
}
"#,
        );
    }

    #[test]
    fn function_use_local_in_param() {
        cov_mark::check!(inline_call_inline_locals);
        check_assist(
            inline_call,
            r#"
fn square(x: u32) -> u32 {
    x * x
}
fn main() {
    let local = 51;
    let y = square$0(local);
}
"#,
            r#"
fn square(x: u32) -> u32 {
    x * x
}
fn main() {
    let local = 51;
    let y = local * local;
}
"#,
        );
    }

    #[test]
    fn method_in_impl() {
        check_assist(
            inline_call,
            r#"
struct Foo;
impl Foo {
    fn foo(&self) {
        self;
        self;
    }
    fn bar(&self) {
        self.foo$0();
    }
}
"#,
            r#"
struct Foo;
impl Foo {
    fn foo(&self) {
        self;
        self;
    }
    fn bar(&self) {
        {
            let ref this = self;
            this;
            this;
        };
    }
}
"#,
        );
    }

    #[test]
    fn wraps_closure_in_paren() {
        cov_mark::check!(inline_call_inline_closure);
        check_assist(
            inline_call,
            r#"
fn foo(x: fn()) {
    x();
}

fn main() {
    foo$0(|| {})
}
"#,
            r#"
fn foo(x: fn()) {
    x();
}

fn main() {
    {
        (|| {})();
    }
}
"#,
        );
        check_assist(
            inline_call,
            r#"
fn foo(x: fn()) {
    x();
}

fn main() {
    foo$0(main)
}
"#,
            r#"
fn foo(x: fn()) {
    x();
}

fn main() {
    {
        main();
    }
}
"#,
        );
    }

    #[test]
    fn inline_single_literal_expr() {
        cov_mark::check!(inline_call_inline_literal);
        check_assist(
            inline_call,
            r#"
fn foo(x: u32) -> u32{
    x
}

fn main() {
    foo$0(222);
}
"#,
            r#"
fn foo(x: u32) -> u32{
    x
}

fn main() {
    222;
}
"#,
        );
    }

    #[test]
    fn inline_emits_type_for_coercion() {
        check_assist(
            inline_call,
            r#"
fn foo(x: *const u32) -> u32 {
    x as u32
}

fn main() {
    foo$0(&222);
}
"#,
            r#"
fn foo(x: *const u32) -> u32 {
    x as u32
}

fn main() {
    {
        let x: *const u32 = &222;
        x as u32
    };
}
"#,
        );
    }

    // FIXME: const generics aren't being substituted, this is blocked on better support for them
    #[test]
    fn inline_substitutes_generics() {
        check_assist(
            inline_call,
            r#"
fn foo<T, const N: usize>() {
    bar::<T, N>()
}

fn bar<U, const M: usize>() {}

fn main() {
    foo::<usize, {0}>$0();
}
"#,
            r#"
fn foo<T, const N: usize>() {
    bar::<T, N>()
}

fn bar<U, const M: usize>() {}

fn main() {
    bar::<usize, N>();
}
"#,
        );
    }
}
