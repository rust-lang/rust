use std::iter::once;

use ide_db::ty_filter::TryEnum;
use syntax::{
    ast::{
        self,
        edit::{AstNodeEdit, IndentLevel},
        make,
    },
    AstNode, T,
};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: replace_let_with_if_let
//
// Replaces `let` with an `if let`.
//
// ```
// # enum Option<T> { Some(T), None }
//
// fn main(action: Action) {
//     $0let x = compute();
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
pub(crate) fn replace_let_with_if_let(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let let_kw = ctx.find_token_syntax_at_offset(T![let])?;
    let let_stmt = let_kw.parent().and_then(ast::LetStmt::cast)?;
    let init = let_stmt.initializer()?;
    let original_pat = let_stmt.pat()?;

    let target = let_kw.text_range();
    acc.add(
        AssistId("replace_let_with_if_let", AssistKind::RefactorRewrite),
        "Replace let with if let",
        target,
        |edit| {
            let ty = ctx.sema.type_of_expr(&init);
            let happy_variant = ty
                .and_then(|ty| TryEnum::from_ty(&ctx.sema, &ty.adjusted()))
                .map(|it| it.happy_case());
            let pat = match happy_variant {
                None => original_pat,
                Some(var_name) => {
                    make::tuple_struct_pat(make::ext::ident_path(var_name), once(original_pat))
                        .into()
                }
            };

            let block =
                make::ext::empty_block_expr().indent(IndentLevel::from_node(let_stmt.syntax()));
            let if_ = make::expr_if(make::expr_let(pat, init).into(), block, None);
            let stmt = make::expr_stmt(if_);

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
    $0let x = E::X(92);
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
