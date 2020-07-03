use ra_syntax::{
    ast::{self, AstNode},
    TextSize, T,
};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: remove_dbg
//
// Removes `dbg!()` macro call.
//
// ```
// fn main() {
//     <|>dbg!(92);
// }
// ```
// ->
// ```
// fn main() {
//     92;
// }
// ```
pub(crate) fn remove_dbg(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let macro_call = ctx.find_node_at_offset::<ast::MacroCall>()?;

    if !is_valid_macrocall(&macro_call, "dbg")? {
        return None;
    }

    let macro_range = macro_call.syntax().text_range();

    let macro_content = {
        let macro_args = macro_call.token_tree()?.syntax().clone();

        let text = macro_args.text();
        let without_parens = TextSize::of('(')..text.len() - TextSize::of(')');
        text.slice(without_parens).to_string()
    };

    let target = macro_call.syntax().text_range();
    acc.add(AssistId("remove_dbg", AssistKind::Refactor), "Remove dbg!()", target, |builder| {
        builder.replace(macro_range, macro_content);
    })
}

/// Verifies that the given macro_call actually matches the given name
/// and contains proper ending tokens
fn is_valid_macrocall(macro_call: &ast::MacroCall, macro_name: &str) -> Option<bool> {
    let path = macro_call.path()?;
    let name_ref = path.segment()?.name_ref()?;

    // Make sure it is actually a dbg-macro call, dbg followed by !
    let excl = path.syntax().next_sibling_or_token()?;

    if name_ref.text() != macro_name || excl.kind() != T![!] {
        return None;
    }

    let node = macro_call.token_tree()?.syntax().clone();
    let first_child = node.first_child_or_token()?;
    let last_child = node.last_child_or_token()?;

    match (first_child.kind(), last_child.kind()) {
        (T!['('], T![')']) | (T!['['], T![']']) | (T!['{'], T!['}']) => Some(true),
        _ => Some(false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{check_assist, check_assist_not_applicable, check_assist_target};

    #[test]
    fn test_remove_dbg() {
        check_assist(remove_dbg, "<|>dbg!(1 + 1)", "1 + 1");

        check_assist(remove_dbg, "dbg!<|>((1 + 1))", "(1 + 1)");

        check_assist(remove_dbg, "dbg!(1 <|>+ 1)", "1 + 1");

        check_assist(remove_dbg, "let _ = <|>dbg!(1 + 1)", "let _ = 1 + 1");

        check_assist(
            remove_dbg,
            "
fn foo(n: usize) {
    if let Some(_) = dbg!(n.<|>checked_sub(4)) {
        // ...
    }
}
",
            "
fn foo(n: usize) {
    if let Some(_) = n.checked_sub(4) {
        // ...
    }
}
",
        );
    }
    #[test]
    fn test_remove_dbg_with_brackets_and_braces() {
        check_assist(remove_dbg, "dbg![<|>1 + 1]", "1 + 1");
        check_assist(remove_dbg, "dbg!{<|>1 + 1}", "1 + 1");
    }

    #[test]
    fn test_remove_dbg_not_applicable() {
        check_assist_not_applicable(remove_dbg, "<|>vec![1, 2, 3]");
        check_assist_not_applicable(remove_dbg, "<|>dbg(5, 6, 7)");
        check_assist_not_applicable(remove_dbg, "<|>dbg!(5, 6, 7");
    }

    #[test]
    fn remove_dbg_target() {
        check_assist_target(
            remove_dbg,
            "
fn foo(n: usize) {
    if let Some(_) = dbg!(n.<|>checked_sub(4)) {
        // ...
    }
}
",
            "dbg!(n.checked_sub(4))",
        );
    }
}
