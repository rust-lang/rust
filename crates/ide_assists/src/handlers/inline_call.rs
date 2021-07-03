use ast::make;
use hir::{HasSource, PathResolution};
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
// Inlines a function or method body.
//
// ```
// fn add(a: u32, b: u32) -> u32 { a + b }
// fn main() {
//     let x = add$0(1, 2);
// }
// ```
// ->
// ```
// fn add(a: u32, b: u32) -> u32 { a + b }
// fn main() {
//     let x = {
//         let a = 1;
//         let b = 2;
//         a + b
//     };
// }
// ```
pub(crate) fn inline_call(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let (label, function, arguments, expr) =
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
                ast::Expr::CallExpr(call),
            )
        } else {
            let name_ref: ast::NameRef = ctx.find_node_at_offset()?;
            let call = name_ref.syntax().parent().and_then(ast::MethodCallExpr::cast)?;
            let receiver = call.receiver()?;
            let function = ctx.sema.resolve_method_call(&call)?;
            let mut arguments = vec![receiver];
            arguments.extend(call.arg_list()?.args());
            (format!("Inline `{}`", name_ref), function, arguments, ast::Expr::MethodCallExpr(call))
        };

    inline_(acc, ctx, label, function, arguments, expr)
}

pub(crate) fn inline_(
    acc: &mut Assists,
    ctx: &AssistContext,
    label: String,
    function: hir::Function,
    arg_list: Vec<ast::Expr>,
    expr: ast::Expr,
) -> Option<()> {
    let hir::InFile { value: function_source, .. } = function.source(ctx.db())?;
    let param_list = function_source.param_list()?;

    let mut params = Vec::new();
    if let Some(self_param) = param_list.self_param() {
        // FIXME this should depend on the receiver as well as the self_param
        params.push(
            make::ident_pat(
                self_param.amp_token().is_some(),
                self_param.mut_token().is_some(),
                make::name("this"),
            )
            .into(),
        );
    }
    for param in param_list.params() {
        params.push(param.pat()?);
    }

    if arg_list.len() != params.len() {
        // Can't inline the function because they've passed the wrong number of
        // arguments to this function
        cov_mark::hit!(inline_call_incorrect_number_of_arguments);
        return None;
    }

    let new_bindings = params.into_iter().zip(arg_list);

    let body = function_source.body()?;

    acc.add(
        AssistId("inline_call", AssistKind::RefactorInline),
        label,
        expr.syntax().text_range(),
        |builder| {
            // FIXME: emit type ascriptions when a coercion happens?
            // FIXME: dont create locals when its not required
            let statements = new_bindings
                .map(|(pattern, value)| make::let_stmt(pattern, Some(value)).into())
                .chain(body.statements());

            let original_indentation = expr.indent_level();
            let mut replacement = make::block_expr(statements, body.tail_expr())
                .reset_indent()
                .indent(original_indentation);

            if param_list.self_param().is_some() {
                replacement = replacement.clone_for_update();
                let this = make::name_ref("this").syntax().clone_for_update();
                // FIXME dont look into descendant methods
                replacement
                    .syntax()
                    .descendants()
                    .filter_map(ast::NameRef::cast)
                    .filter(|n| n.self_token().is_some())
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .for_each(|self_ref| ted::replace(self_ref.syntax(), &this));
            }
            builder.replace_ast(expr, ast::Expr::BlockExpr(replacement));
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
    {
        println!("Hello, World!");
    };
}
"#,
        );
    }

    #[test]
    fn args_with_side_effects() {
        check_assist(
            inline_call,
            r#"
fn foo(name: String) { println!("Hello, {}!", name); }
fn main() {
    foo$0(String::from("Michael"));
}
"#,
            r#"
fn foo(name: String) { println!("Hello, {}!", name); }
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
        let a = 1;
        let b = 2;
        let x = a + b;
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
        let a = 2;
        Foo(this.0 + a)
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
        let a = 2;
        Foo(this.0 + a)
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
        let a = 2;
        Foo(this.0 + a)
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
}
