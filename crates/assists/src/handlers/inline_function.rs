use ast::make;
use hir::{HasSource, PathResolution};
use syntax::{
    ast::{self, edit::AstNodeEdit, ArgListOwner},
    AstNode,
};
use test_utils::mark;

use crate::{
    assist_context::{AssistContext, Assists},
    AssistId, AssistKind,
};

// Assist: inline_function
//
// Inlines a function body.
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
pub(crate) fn inline_function(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let path_expr: ast::PathExpr = ctx.find_node_at_offset()?;
    let call = path_expr.syntax().parent().and_then(ast::CallExpr::cast)?;
    let path = path_expr.path()?;

    let function = match ctx.sema.resolve_path(&path)? {
        PathResolution::Def(hir::ModuleDef::Function(f)) => f,
        _ => return None,
    };

    let function_source = function.source(ctx.db())?;
    let arguments: Vec<_> = call.arg_list()?.args().collect();
    let parameters = function_parameter_patterns(&function_source.value)?;

    if arguments.len() != parameters.len() {
        // Can't inline the function because they've passed the wrong number of
        // arguments to this function
        mark::hit!(inline_function_incorrect_number_of_arguments);
        return None;
    }

    let new_bindings = parameters.into_iter().zip(arguments);

    let body = function_source.value.body()?;

    acc.add(
        AssistId("inline_function", AssistKind::RefactorInline),
        format!("Inline `{}`", path),
        call.syntax().text_range(),
        |builder| {
            let mut statements: Vec<ast::Stmt> = Vec::new();

            for (pattern, value) in new_bindings {
                statements.push(make::let_stmt(pattern, Some(value)).into());
            }

            statements.extend(body.statements());

            let original_indentation = call.indent_level();
            let replacement = make::block_expr(statements, body.tail_expr())
                .reset_indent()
                .indent(original_indentation);

            builder.replace_ast(ast::Expr::CallExpr(call), ast::Expr::BlockExpr(replacement));
        },
    )
}

fn function_parameter_patterns(value: &ast::Fn) -> Option<Vec<ast::Pat>> {
    let mut patterns = Vec::new();

    for param in value.param_list()?.params() {
        let pattern = param.pat()?;
        patterns.push(pattern);
    }

    Some(patterns)
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn no_args_or_return_value_gets_inlined_without_block() {
        check_assist(
            inline_function,
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
            inline_function,
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
    fn method_inlining_isnt_supported() {
        check_assist_not_applicable(
            inline_function,
            r"
struct Foo;
impl Foo { fn bar(&self) {} }

fn main() { Foo.bar$0(); }
",
        );
    }

    #[test]
    fn not_applicable_when_incorrect_number_of_parameters_are_provided() {
        mark::check!(inline_function_incorrect_number_of_arguments);
        check_assist_not_applicable(
            inline_function,
            r#"
fn add(a: u32, b: u32) -> u32 { a + b }
fn main() { let x = add$0(42); }
"#,
        );
    }

    #[test]
    fn function_with_multiple_statements() {
        check_assist(
            inline_function,
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
}
