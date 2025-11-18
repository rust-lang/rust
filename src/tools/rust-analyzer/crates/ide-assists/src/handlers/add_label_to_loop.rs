use ide_db::{
    source_change::SourceChangeBuilder, syntax_helpers::node_ext::for_each_break_and_continue_expr,
};
use syntax::{
    SyntaxToken, T,
    ast::{
        self, AstNode, HasLoopBody,
        make::{self, tokens},
        syntax_factory::SyntaxFactory,
    },
    syntax_editor::{Position, SyntaxEditor},
};

use crate::{AssistContext, AssistId, Assists};

// Assist: add_label_to_loop
//
// Adds a label to a loop.
//
// ```
// fn main() {
//     loop$0 {
//         break;
//         continue;
//     }
// }
// ```
// ->
// ```
// fn main() {
//     ${1:'l}: loop {
//         break ${2:'l};
//         continue ${0:'l};
//     }
// }
// ```
pub(crate) fn add_label_to_loop(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let loop_kw = ctx.find_token_syntax_at_offset(T![loop])?;
    let loop_expr = loop_kw.parent().and_then(ast::LoopExpr::cast)?;
    if loop_expr.label().is_some() {
        return None;
    }

    acc.add(
        AssistId::generate("add_label_to_loop"),
        "Add Label",
        loop_expr.syntax().text_range(),
        |builder| {
            let make = SyntaxFactory::with_mappings();
            let mut editor = builder.make_editor(loop_expr.syntax());

            let label = make.lifetime("'l");
            let elements = vec![
                label.syntax().clone().into(),
                make::token(T![:]).into(),
                tokens::single_space().into(),
            ];
            editor.insert_all(Position::before(&loop_kw), elements);

            if let Some(cap) = ctx.config.snippet_cap {
                editor.add_annotation(label.syntax(), builder.make_placeholder_snippet(cap));
            }

            let loop_body = loop_expr.loop_body().and_then(|it| it.stmt_list());
            for_each_break_and_continue_expr(loop_expr.label(), loop_body, &mut |expr| {
                let token = match expr {
                    ast::Expr::BreakExpr(break_expr) => break_expr.break_token(),
                    ast::Expr::ContinueExpr(continue_expr) => continue_expr.continue_token(),
                    _ => return,
                };
                if let Some(token) = token {
                    insert_label_after_token(&mut editor, &make, &token, ctx, builder);
                }
            });

            editor.add_mappings(make.finish_with_mappings());
            builder.add_file_edits(ctx.vfs_file_id(), editor);
            builder.rename();
        },
    )
}

fn insert_label_after_token(
    editor: &mut SyntaxEditor,
    make: &SyntaxFactory,
    token: &SyntaxToken,
    ctx: &AssistContext<'_>,
    builder: &mut SourceChangeBuilder,
) {
    let label = make.lifetime("'l");
    let elements = vec![tokens::single_space().into(), label.syntax().clone().into()];
    editor.insert_all(Position::after(token), elements);

    if let Some(cap) = ctx.config.snippet_cap {
        editor.add_annotation(label.syntax(), builder.make_placeholder_snippet(cap));
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn add_label() {
        check_assist(
            add_label_to_loop,
            r#"
fn main() {
    loop$0 {
        break;
        continue;
    }
}"#,
            r#"
fn main() {
    ${1:'l}: loop {
        break ${2:'l};
        continue ${0:'l};
    }
}"#,
        );
    }

    #[test]
    fn add_label_to_outer_loop() {
        check_assist(
            add_label_to_loop,
            r#"
fn main() {
    loop$0 {
        break;
        continue;
        loop {
            break;
            continue;
        }
    }
}"#,
            r#"
fn main() {
    ${1:'l}: loop {
        break ${2:'l};
        continue ${0:'l};
        loop {
            break;
            continue;
        }
    }
}"#,
        );
    }

    #[test]
    fn add_label_to_inner_loop() {
        check_assist(
            add_label_to_loop,
            r#"
fn main() {
    loop {
        break;
        continue;
        loop$0 {
            break;
            continue;
        }
    }
}"#,
            r#"
fn main() {
    loop {
        break;
        continue;
        ${1:'l}: loop {
            break ${2:'l};
            continue ${0:'l};
        }
    }
}"#,
        );
    }

    #[test]
    fn do_not_add_label_if_exists() {
        check_assist_not_applicable(
            add_label_to_loop,
            r#"
fn main() {
    'l: loop$0 {
        break 'l;
        continue 'l;
    }
}"#,
        );
    }
}
