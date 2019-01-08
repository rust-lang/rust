use ra_syntax::{
    Direction,
    SyntaxKind::COMMA,
};

use crate::assists::{non_trivia_sibling, AssistCtx, Assist};

pub fn flip_comma(ctx: AssistCtx) -> Option<Assist> {
    let comma = ctx.leaf_at_offset().find(|leaf| leaf.kind() == COMMA)?;
    let prev = non_trivia_sibling(comma, Direction::Prev)?;
    let next = non_trivia_sibling(comma, Direction::Next)?;
    ctx.build("flip comma", |edit| {
        edit.replace(prev.range(), next.text());
        edit.replace(next.range(), prev.text());
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assists::check_assist;

    #[test]
    fn flip_comma_works_for_function_parameters() {
        check_assist(
            flip_comma,
            "fn foo(x: i32,<|> y: Result<(), ()>) {}",
            "fn foo(y: Result<(), ()>,<|> x: i32) {}",
        )
    }
}
