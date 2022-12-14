use syntax::{SyntaxKind, TextRange, T};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: remove_mut
//
// Removes the `mut` keyword.
//
// ```
// impl Walrus {
//     fn feed(&mut$0 self, amount: u32) {}
// }
// ```
// ->
// ```
// impl Walrus {
//     fn feed(&self, amount: u32) {}
// }
// ```
pub(crate) fn remove_mut(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let mut_token = ctx.find_token_syntax_at_offset(T![mut])?;
    let delete_from = mut_token.text_range().start();
    let delete_to = match mut_token.next_token() {
        Some(it) if it.kind() == SyntaxKind::WHITESPACE => it.text_range().end(),
        _ => mut_token.text_range().end(),
    };

    let target = mut_token.text_range();
    acc.add(
        AssistId("remove_mut", AssistKind::Refactor),
        "Remove `mut` keyword",
        target,
        |builder| {
            builder.delete(TextRange::new(delete_from, delete_to));
        },
    )
}
