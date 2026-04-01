use syntax::{AstNode, SyntaxKind, T, TextRange, ast};

use crate::{AssistContext, AssistId, Assists};

// Assist: remove_else_branches
//
// Removes the `else` keyword and else branches.
//
// ```
// fn main() {
//     if true {
//         let _ = 2;
//     } $0else {
//         unreachable!();
//     }
// }
// ```
// ->
// ```
// fn main() {
//     if true {
//         let _ = 2;
//     }
// }
// ```
// ---
// ```
// fn main() {
//     let _x = 2 $0else { unreachable!() };
// }
// ```
// ->
// ```
// fn main() {
//     let _x = 2;
// }
// ```
pub(crate) fn remove_else_branches(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let else_token = ctx.find_token_syntax_at_offset(T![else])?;
    let else_branches = ctx
        .find_node_at_range::<ast::IfExpr>()
        .and_then(|if_expr| if_expr.else_branch()?.syntax().clone().into())
        .or_else(|| {
            ctx.find_node_at_range::<ast::LetStmt>()?
                .let_else()?
                .block_expr()?
                .syntax()
                .clone()
                .into()
        })?;

    let target = TextRange::cover(else_token.text_range(), else_branches.text_range());
    acc.add(
        AssistId::refactor("remove_else_branches"),
        "Remove `else` branches",
        target,
        |builder| {
            let mut editor = builder.make_editor(&else_token.parent().unwrap());
            match else_token.prev_token() {
                Some(it) if it.kind() == SyntaxKind::WHITESPACE => editor.delete(it),
                _ => (),
            }
            match else_token.next_token() {
                Some(it) if it.kind() == SyntaxKind::WHITESPACE => editor.delete(it),
                _ => (),
            }
            editor.delete(else_token);
            editor.delete(else_branches);
            builder.add_file_edits(ctx.vfs_file_id(), editor);
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::check_assist_not_applicable;

    #[test]
    fn test_remove_else_branches_not_on_else_token() {
        check_assist_not_applicable(
            remove_else_branches,
            r#"
fn main() {
    let _x = 2 else {$0 unreachable!() };
}
"#,
        );
    }
}
