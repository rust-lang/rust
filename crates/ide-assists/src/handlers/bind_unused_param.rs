use crate::assist_context::{AssistContext, Assists};
use ide_db::{
    assists::{AssistId, AssistKind},
    defs::Definition,
    LineIndexDatabase,
};
use syntax::{
    ast::{self, edit_in_place::Indent},
    AstNode,
};

use super::remove_unused_param::is_trait_impl;

// Assist: bind_unused_param
//
// Binds unused function parameter to an underscore.
//
// ```
// fn some_function(x: i32$0) {}
// ```
// ->
// ```
// fn some_function(x: i32) {
//     let _ = x;
// }
// ```
pub(crate) fn bind_unused_param(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let param: ast::Param = ctx.find_node_at_offset()?;

    let Some(ast::Pat::IdentPat(ident_pat)) = param.pat() else { return None };

    let func = param.syntax().ancestors().find_map(ast::Fn::cast)?;
    if is_trait_impl(&func) {
        cov_mark::hit!(trait_impl);
        return None;
    }

    let param_def = {
        let local = ctx.sema.to_def(&ident_pat)?;
        Definition::Local(local)
    };
    if param_def.usages(&ctx.sema).at_least_one() {
        cov_mark::hit!(keep_used);
        return None;
    }

    let stmt_list = func.body()?.stmt_list()?;
    let l_curly_range = stmt_list.l_curly_token()?.text_range();
    let r_curly_range = stmt_list.r_curly_token()?.text_range();

    acc.add(
        AssistId("bind_unused_param", AssistKind::QuickFix),
        &format!("Bind as `let _ = {};`", &ident_pat),
        param.syntax().text_range(),
        |builder| {
            let line_index = ctx.db().line_index(ctx.file_id());

            let mut indent = func.indent_level();
            indent.0 += 1;
            let mut text = format!("\n{indent}let _ = {ident_pat};");

            let left_line = line_index.line_col(l_curly_range.end()).line;
            let right_line = line_index.line_col(r_curly_range.start()).line;

            if left_line == right_line {
                cov_mark::hit!(single_line);
                text.push('\n');
            }

            builder.insert(l_curly_range.end(), text);
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn bind_unused_empty_block() {
        cov_mark::check!(single_line);
        check_assist(
            bind_unused_param,
            r#"
fn foo($0y: i32) {}
"#,
            r#"
fn foo(y: i32) {
    let _ = y;
}
"#,
        );
    }

    #[test]
    fn bind_unused_empty_block_with_newline() {
        check_assist(
            bind_unused_param,
            r#"
fn foo($0y: i32) {
}
"#,
            r#"
fn foo(y: i32) {
    let _ = y;
}
"#,
        );
    }

    #[test]
    fn bind_unused_generic() {
        check_assist(
            bind_unused_param,
            r#"
fn foo<T>($0y: T)
where T : Default {
}
"#,
            r#"
fn foo<T>(y: T)
where T : Default {
    let _ = y;
}
"#,
        );
    }

    #[test]
    fn trait_impl() {
        cov_mark::check!(trait_impl);
        check_assist_not_applicable(
            bind_unused_param,
            r#"
trait Trait {
    fn foo(x: i32);
}
impl Trait for () {
    fn foo($0x: i32) {}
}
"#,
        );
    }

    #[test]
    fn keep_used() {
        cov_mark::check!(keep_used);
        check_assist_not_applicable(
            bind_unused_param,
            r#"
fn foo(x: i32, $0y: i32) { y; }
"#,
        );
    }
}
