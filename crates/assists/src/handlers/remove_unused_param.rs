use ide_db::{base_db::FileId, defs::Definition, search::FileReference};
use syntax::{
    algo::find_node_at_range,
    ast::{self, ArgListOwner},
    AstNode, SourceFile, SyntaxKind, SyntaxNode, TextRange, T,
};
use test_utils::mark;
use SyntaxKind::WHITESPACE;

use crate::{
    assist_context::AssistBuilder, utils::next_prev, AssistContext, AssistId, AssistKind, Assists,
};

// Assist: remove_unused_param
//
// Removes unused function parameter.
//
// ```
// fn frobnicate(x: i32$0) {}
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
            builder.delete(range_to_remove(param.syntax()));
            for (file_id, references) in fn_def.usages(&ctx.sema).all() {
                process_usages(ctx, builder, file_id, references, param_position);
            }
        },
    )
}

fn process_usages(
    ctx: &AssistContext,
    builder: &mut AssistBuilder,
    file_id: FileId,
    references: Vec<FileReference>,
    arg_to_remove: usize,
) {
    let source_file = ctx.sema.parse(file_id);
    builder.edit_file(file_id);
    for usage in references {
        if let Some(text_range) = process_usage(&source_file, usage, arg_to_remove) {
            builder.delete(text_range);
        }
    }
}

fn process_usage(
    source_file: &SourceFile,
    FileReference { range, .. }: FileReference,
    arg_to_remove: usize,
) -> Option<TextRange> {
    let call_expr: ast::CallExpr = find_node_at_range(source_file.syntax(), range)?;
    let call_expr_range = call_expr.expr()?.syntax().text_range();
    if !call_expr_range.contains_range(range) {
        return None;
    }
    let arg = call_expr.arg_list()?.args().nth(arg_to_remove)?;
    Some(range_to_remove(arg.syntax()))
}

fn range_to_remove(node: &SyntaxNode) -> TextRange {
    let up_to_comma = next_prev().find_map(|dir| {
        node.siblings_with_tokens(dir)
            .filter_map(|it| it.into_token())
            .find(|it| it.kind() == T![,])
            .map(|it| (dir, it))
    });
    if let Some((dir, token)) = up_to_comma {
        if node.next_sibling().is_some() {
            let up_to_space = token
                .siblings_with_tokens(dir)
                .skip(1)
                .take_while(|it| it.kind() == WHITESPACE)
                .last()
                .and_then(|it| it.into_token());
            return node
                .text_range()
                .cover(up_to_space.map_or(token.text_range(), |it| it.text_range()));
        }
        node.text_range().cover(token.text_range())
    } else {
        node.text_range()
    }
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
fn foo(x: i32, $0y: i32) { x; }
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
    fn remove_unused_first_param() {
        check_assist(
            remove_unused_param,
            r#"
fn foo($0x: i32, y: i32) { y; }
fn a() { foo(1, 2) }
fn b() { foo(1, 2,) }
"#,
            r#"
fn foo(y: i32) { y; }
fn a() { foo(2) }
fn b() { foo(2,) }
"#,
        );
    }

    #[test]
    fn remove_unused_single_param() {
        check_assist(
            remove_unused_param,
            r#"
fn foo($0x: i32) { 0; }
fn a() { foo(1) }
fn b() { foo(1, ) }
"#,
            r#"
fn foo() { 0; }
fn a() { foo() }
fn b() { foo( ) }
"#,
        );
    }

    #[test]
    fn remove_unused_surrounded_by_parms() {
        check_assist(
            remove_unused_param,
            r#"
fn foo(x: i32, $0y: i32, z: i32) { x; }
fn a() { foo(1, 2, 3) }
fn b() { foo(1, 2, 3,) }
"#,
            r#"
fn foo(x: i32, z: i32) { x; }
fn a() { foo(1, 3) }
fn b() { foo(1, 3,) }
"#,
        );
    }

    #[test]
    fn remove_unused_qualified_call() {
        check_assist(
            remove_unused_param,
            r#"
mod bar { pub fn foo(x: i32, $0y: i32) { x; } }
fn b() { bar::foo(9, 2) }
"#,
            r#"
mod bar { pub fn foo(x: i32) { x; } }
fn b() { bar::foo(9) }
"#,
        );
    }

    #[test]
    fn remove_unused_turbofished_func() {
        check_assist(
            remove_unused_param,
            r#"
pub fn foo<T>(x: T, $0y: i32) { x; }
fn b() { foo::<i32>(9, 2) }
"#,
            r#"
pub fn foo<T>(x: T) { x; }
fn b() { foo::<i32>(9) }
"#,
        );
    }

    #[test]
    fn remove_unused_generic_unused_param_func() {
        check_assist(
            remove_unused_param,
            r#"
pub fn foo<T>(x: i32, $0y: T) { x; }
fn b() { foo::<i32>(9, 2) }
fn b2() { foo(9, 2) }
"#,
            r#"
pub fn foo<T>(x: i32) { x; }
fn b() { foo::<i32>(9) }
fn b2() { foo(9) }
"#,
        );
    }

    #[test]
    fn keep_used() {
        mark::check!(keep_used);
        check_assist_not_applicable(
            remove_unused_param,
            r#"
fn foo(x: i32, $0y: i32) { y; }
fn main() { foo(9, 2) }
"#,
        );
    }

    #[test]
    fn remove_across_files() {
        check_assist(
            remove_unused_param,
            r#"
//- /main.rs
fn foo(x: i32, $0y: i32) { x; }

mod foo;

//- /foo.rs
use super::foo;

fn bar() {
    let _ = foo(1, 2);
}
"#,
            r#"
//- /main.rs
fn foo(x: i32) { x; }

mod foo;

//- /foo.rs
use super::foo;

fn bar() {
    let _ = foo(1);
}
"#,
        )
    }
}
