use ide_db::syntax_helpers::node_ext::for_each_break_and_continue_expr;
use syntax::{
    SyntaxToken, T,
    ast::{self, AstNode, HasLoopBody},
    syntax_editor::{Position, SyntaxAnnotation, SyntaxEditor},
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
//     ${0:'l}: loop {
//         break ${0:'l};
//         continue ${0:'l};
//     }
// }
// ```
pub(crate) fn add_label_to_loop(acc: &mut Assists, ctx: &AssistContext<'_, '_>) -> Option<()> {
    let loop_expr = ctx.find_node_at_offset::<ast::AnyHasLoopBody>()?;
    let loop_kw = loop_token(&loop_expr)?;
    if loop_expr.label().is_some() || !loop_kw.text_range().contains_inclusive(ctx.offset()) {
        return None;
    }

    acc.add(
        AssistId::generate("add_label_to_loop"),
        "Add Label",
        loop_expr.syntax().text_range(),
        |builder| {
            let editor = builder.make_editor(loop_expr.syntax());
            let make = editor.make();

            let label = make.lifetime("'l");
            let elements = vec![
                label.syntax().clone().into(),
                make.token(T![:]).into(),
                make.whitespace(" ").into(),
            ];
            editor.insert_all(Position::before(&loop_kw), elements);

            let annotation =
                ctx.config.snippet_cap.map(|cap| builder.make_placeholder_snippet(cap));

            if let Some(annotation) = annotation {
                editor.add_annotation(label.syntax(), annotation);
            }

            let loop_body = loop_expr.loop_body().and_then(|it| it.stmt_list());
            for_each_break_and_continue_expr(loop_expr.label(), loop_body, &mut |expr| {
                let token = match expr {
                    ast::Expr::BreakExpr(break_expr) => break_expr.break_token(),
                    ast::Expr::ContinueExpr(continue_expr) => continue_expr.continue_token(),
                    _ => return,
                };
                if let Some(token) = token
                    && let Some(annotation) = annotation
                {
                    insert_label_after_token(&editor, &token, annotation);
                }
            });

            builder.add_file_edits(ctx.vfs_file_id(), editor);
        },
    )
}

fn loop_token(loop_expr: &ast::AnyHasLoopBody) -> Option<syntax::SyntaxToken> {
    loop_expr
        .syntax()
        .children_with_tokens()
        .filter_map(|it| it.into_token())
        .find(|it| matches!(it.kind(), T![for] | T![loop] | T![while]))
}

fn insert_label_after_token(
    editor: &SyntaxEditor,
    token: &SyntaxToken,
    annotation: SyntaxAnnotation,
) {
    let make = editor.make();
    let label = make.lifetime("'l");
    let elements = vec![make.whitespace(" ").into(), label.syntax().clone().into()];
    editor.insert_all(Position::after(token), elements);

    editor.add_annotation(label.syntax(), annotation);
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
    ${0:'l}: loop {
        break ${0:'l};
        continue ${0:'l};
    }
}"#,
        );
    }

    #[test]
    fn add_label_to_while_expr() {
        check_assist(
            add_label_to_loop,
            r#"
fn main() {
    while$0 true {
        break;
        continue;
    }
}"#,
            r#"
fn main() {
    ${0:'l}: while true {
        break ${0:'l};
        continue ${0:'l};
    }
}"#,
        );
    }

    #[test]
    fn add_label_to_for_expr() {
        check_assist(
            add_label_to_loop,
            r#"
fn main() {
    for$0 _ in 0..5 {
        break;
        continue;
    }
}"#,
            r#"
fn main() {
    ${0:'l}: for _ in 0..5 {
        break ${0:'l};
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
    ${0:'l}: loop {
        break ${0:'l};
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
        ${0:'l}: loop {
            break ${0:'l};
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

    #[test]
    fn do_not_add_label_if_outside_keyword() {
        check_assist_not_applicable(
            add_label_to_loop,
            r#"
fn main() {
    'l: loop {$0
        break 'l;
        continue 'l;
    }
}"#,
        );

        check_assist_not_applicable(
            add_label_to_loop,
            r#"
fn main() {
    'l: while true {$0
        break 'l;
        continue 'l;
    }
}"#,
        );
    }
}
