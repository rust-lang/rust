use ra_syntax::{SyntaxKind, TextRange, T};

use crate::{Assist, AssistCtx, AssistId};

// Assist: remove_mut
//
// Removes the `mut` keyword.
//
// ```
// impl Walrus {
//     fn feed(&mut<|> self, amount: u32) {}
// }
// ```
// ->
// ```
// impl Walrus {
//     fn feed(&self, amount: u32) {}
// }
// ```
pub(crate) fn remove_mut(ctx: AssistCtx) -> Option<Assist> {
    let mut_token = ctx.find_token_at_offset(T![mut])?;
    let delete_from = mut_token.text_range().start();
    let delete_to = match mut_token.next_token() {
        Some(it) if it.kind() == SyntaxKind::WHITESPACE => it.text_range().end(),
        _ => mut_token.text_range().end(),
    };

    ctx.add_assist(AssistId("remove_mut"), "Remove `mut` keyword", |edit| {
        edit.set_cursor(delete_from);
        edit.delete(TextRange::from_to(delete_from, delete_to));
    })
}
