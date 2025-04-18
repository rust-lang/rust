use syntax::{SyntaxKind, T};

use crate::{AssistContext, AssistId, Assists};

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

    let target = mut_token.text_range();
    acc.add(AssistId::refactor("remove_mut"), "Remove `mut` keyword", target, |builder| {
        let mut editor = builder.make_editor(&mut_token.parent().unwrap());
        match mut_token.next_token() {
            Some(it) if it.kind() == SyntaxKind::WHITESPACE => editor.delete(it),
            _ => (),
        }
        editor.delete(mut_token);
        builder.add_file_edits(ctx.vfs_file_id(), editor);
    })
}
