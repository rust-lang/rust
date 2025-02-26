use ide_db::ty_filter::TryEnum;
use syntax::{
    ast::{self, edit::IndentLevel, edit_in_place::Indent, syntax_factory::SyntaxFactory},
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
        |builder| {
            let mut editor = builder.make_editor(let_stmt.syntax());
            let make = SyntaxFactory::new();
            let ty = ctx.sema.type_of_expr(&init);
            let happy_variant = ty
                .and_then(|ty| TryEnum::from_ty(&ctx.sema, &ty.adjusted()))
                .map(|it| it.happy_case());
            let pat = match happy_variant {
                None => original_pat,
                Some(var_name) => {
                    make.tuple_struct_pat(make.ident_path(var_name), [original_pat]).into()
                }
            };

            let block = make.block_expr([], None);
            block.indent(IndentLevel::from_node(let_stmt.syntax()));
            let if_expr = make.expr_if(make.expr_let(pat, init).into(), block, None);
            let if_stmt = make.expr_stmt(if_expr.into());

            editor.replace(let_stmt.syntax(), if_stmt.syntax());
            editor.add_mappings(make.finish_with_mappings());
            builder.add_file_edits(ctx.file_id(), editor);
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
