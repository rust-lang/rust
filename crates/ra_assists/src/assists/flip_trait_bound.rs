//! Assist for swapping traits inside of a trait bound list
//!
//! E.g. `A + B` => `B + A` when the cursor is placed by the `+` inside of a
//! trait bound list

use hir::db::HirDatabase;
use ra_syntax::{algo::non_trivia_sibling, ast::TypeBoundList, Direction, T};

use crate::{Assist, AssistCtx, AssistId};

/// Flip trait bound assist.
pub(crate) fn flip_trait_bound(mut ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    // Make sure we're in a `TypeBoundList`
    ctx.node_at_offset::<TypeBoundList>()?;

    // We want to replicate the behavior of `flip_binexpr` by only suggesting
    // the assist when the cursor is on a `+`
    let plus = ctx.token_at_offset().find(|tkn| tkn.kind() == T![+])?;

    let (before, after) = (
        non_trivia_sibling(plus.clone().into(), Direction::Prev)?,
        non_trivia_sibling(plus.clone().into(), Direction::Next)?,
    );

    ctx.add_action(AssistId("flip_trait_bound"), "flip trait bound", |edit| {
        edit.target(plus.text_range());
        edit.replace(before.text_range(), after.to_string());
        edit.replace(after.text_range(), before.to_string());
    });

    ctx.build()
}
