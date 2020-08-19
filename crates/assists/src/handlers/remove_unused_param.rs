use ide_db::{defs::Definition, search::Reference};
use syntax::{
    algo::find_node_at_range,
    ast::{self, ArgListOwner},
    AstNode, SyntaxNode, TextRange, T,
};
use test_utils::mark;

use crate::{
    assist_context::AssistBuilder, utils::next_prev, AssistContext, AssistId, AssistKind, Assists,
};

// Assist: remove_unused_param
//
// Removes unused function parameter.
//
// ```
// fn frobnicate(x: i32<|>) {}
//
// fn main() {
//     frobnicate(92);
// }
// ```
// ->
// ```
// fn frobnicate() {}
//
// fn main() {
//     frobnicate();
// }
// ```
pub(crate) fn remove_unused_param(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let param: ast::Param = ctx.find_node_at_offset()?;
    let ident_pat = match param.pat()? {
        ast::Pat::IdentPat(it) => it,
        _ => return None,
    };
    let func = param.syntax().ancestors().find_map(ast::Fn::cast)?;
    let param_position = func.param_list()?.params().position(|it| it == param)?;

    let fn_def = {
        let func = ctx.sema.to_def(&func)?;
        Definition::ModuleDef(func.into())
    };

    let param_def = {
        let local = ctx.sema.to_def(&ident_pat)?;
        Definition::Local(local)
    };
    if param_def.usages(&ctx.sema).at_least_one() {
        mark::hit!(keep_used);
        return None;
    }
    acc.add(
        AssistId("remove_unused_param", AssistKind::Refactor),
        "Remove unused parameter",
        param.syntax().text_range(),
        |builder| {
            builder.delete(range_with_coma(param.syntax()));
            for usage in fn_def.usages(&ctx.sema).all() {
                process_usage(ctx, builder, usage, param_position);
            }
        },
    )
}

fn process_usage(
    ctx: &AssistContext,
    builder: &mut AssistBuilder,
    usage: Reference,
    arg_to_remove: usize,
) -> Option<()> {
    let source_file = ctx.sema.parse(usage.file_range.file_id);
    let call_expr: ast::CallExpr =
        find_node_at_range(source_file.syntax(), usage.file_range.range)?;
    if call_expr.expr()?.syntax().text_range() != usage.file_range.range {
        return None;
    }
    let arg = call_expr.arg_list()?.args().nth(arg_to_remove)?;

    builder.edit_file(usage.file_range.file_id);
    builder.delete(range_with_coma(arg.syntax()));

    Some(())
}

fn range_with_coma(node: &SyntaxNode) -> TextRange {
    let up_to = next_prev().find_map(|dir| {
        node.siblings_with_tokens(dir)
            .filter_map(|it| it.into_token())
            .find(|it| it.kind() == T![,])
    });
    let up_to = up_to.map_or(node.text_range(), |it| it.text_range());
    node.text_range().cover(up_to)
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn remove_unused() {
        check_assist(
            remove_unused_param,
            r#"
fn a() { foo(9, 2) }
fn foo(x: i32, <|>y: i32) { x; }
fn b() { foo(9, 2,) }
"#,
            r#"
fn a() { foo(9) }
fn foo(x: i32) { x; }
fn b() { foo(9, ) }
"#,
        );
    }

    #[test]
    fn keep_used() {
        mark::check!(keep_used);
        check_assist_not_applicable(
            remove_unused_param,
            r#"
fn foo(x: i32, <|>y: i32) { y; }
fn main() { foo(9, 2) }
"#,
        );
    }
}
