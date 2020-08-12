use std::iter::once;

use syntax::{
    ast::{
        self,
        edit::{AstNodeEdit, IndentLevel},
        make,
    },
    AstNode, T,
};

use crate::{utils::TryEnum, AssistContext, AssistId, AssistKind, Assists};

// Assist: replace_let_with_if_let
//
// Replaces `let` with an `if-let`.
//
// ```
// # enum Option<T> { Some(T), None }
//
// fn main(action: Action) {
//     <|>let x = compute();
// }
//
// fn compute() -> Option<i32> { None }
// ```
// ->
// ```
// # enum Option<T> { Some(T), None }
//
// fn main(action: Action) {
//     if let Some(x) = compute() {
//     }
// }
//
// fn compute() -> Option<i32> { None }
// ```
pub(crate) fn replace_let_with_if_let(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let let_kw = ctx.find_token_at_offset(T![let])?;
    let let_stmt = let_kw.ancestors().find_map(ast::LetStmt::cast)?;
    let init = let_stmt.initializer()?;
    let original_pat = let_stmt.pat()?;
    let ty = ctx.sema.type_of_expr(&init)?;
    let happy_variant = TryEnum::from_ty(&ctx.sema, &ty).map(|it| it.happy_case());

    let target = let_kw.text_range();
    acc.add(
        AssistId("replace_let_with_if_let", AssistKind::RefactorRewrite),
        "Replace with if-let",
        target,
        |edit| {
            let with_placeholder: ast::Pat = match happy_variant {
                None => make::wildcard_pat().into(),
                Some(var_name) => make::tuple_struct_pat(
                    make::path_unqualified(make::path_segment(make::name_ref(var_name))),
                    once(make::wildcard_pat().into()),
                )
                .into(),
            };
            let block =
                make::block_expr(None, None).indent(IndentLevel::from_node(let_stmt.syntax()));
            let if_ = make::expr_if(make::condition(init, Some(with_placeholder)), block);
            let stmt = make::expr_stmt(if_);

            let placeholder = stmt.syntax().descendants().find_map(ast::WildcardPat::cast).unwrap();
            let stmt = stmt.replace_descendant(placeholder.into(), original_pat);

            edit.replace_ast(ast::Stmt::from(let_stmt), ast::Stmt::from(stmt));
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::check_assist;

    use super::*;

    #[test]
    fn replace_let_unknown_enum() {
        check_assist(
            replace_let_with_if_let,
            r"
enum E<T> { X(T), Y(T) }

fn main() {
    <|>let x = E::X(92);
}
            ",
            r"
enum E<T> { X(T), Y(T) }

fn main() {
    if let x = E::X(92) {
    }
}
            ",
        )
    }
}
