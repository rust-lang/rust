use hir::db::HirDatabase;
use ra_syntax::{
    T,
    Direction,
    algo::non_trivia_sibling,
};

use crate::{AssistCtx, Assist, AssistId};

pub(crate) fn flip_comma(mut ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let comma = ctx.token_at_offset().find(|leaf| leaf.kind() == T![,])?;
    let prev = non_trivia_sibling(comma.into(), Direction::Prev)?;
    let next = non_trivia_sibling(comma.into(), Direction::Next)?;
    ctx.add_action(AssistId("flip_comma"), "flip comma", |edit| {
        edit.target(comma.range());
        edit.replace(prev.range(), next.to_string());
        edit.replace(next.range(), prev.to_string());
    });

    ctx.build()
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::helpers::{check_assist, check_assist_target};

    #[test]
    fn flip_comma_works_for_function_parameters() {
        check_assist(
            flip_comma,
            "fn foo(x: i32,<|> y: Result<(), ()>) {}",
            "fn foo(y: Result<(), ()>,<|> x: i32) {}",
        )
    }

    #[test]
    fn flip_comma_target() {
        check_assist_target(flip_comma, "fn foo(x: i32,<|> y: Result<(), ()>) {}", ",")
    }
}
