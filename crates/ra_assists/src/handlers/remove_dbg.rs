use ra_syntax::{
    ast::{self, AstNode},
    TextRange, TextSize, T,
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

    let semicolon_on_end = macro_call.semicolon_token().is_some();
    let is_leaf = macro_call.syntax().next_sibling().is_none();

    let macro_end = match semicolon_on_end {
        true => macro_call.syntax().text_range().end() - TextSize::of(';'),
        false => macro_call.syntax().text_range().end(),
    };

    // macro_range determines what will be deleted and replaced with macro_content
    let macro_range = TextRange::new(macro_call.syntax().text_range().start(), macro_end);
    let paste_instead_of_dbg = {
        let text = macro_call.token_tree()?.syntax().text();

        // leafines determines if we should include the parenthesis or not
        let slice_index: TextRange = match is_leaf {
            // leaf means - we can extract the contents of the dbg! in text
            true => TextRange::new(TextSize::of('('), text.len() - TextSize::of(')')),
            // not leaf - means we should keep the parens
            false => TextRange::new(TextSize::from(0 as u32), text.len()),
        };
        text.slice(slice_index).to_string()
    };

    let target = macro_call.syntax().text_range();
    acc.add(AssistId("remove_dbg", AssistKind::Refactor), "Remove dbg!()", target, |builder| {
        builder.replace(macro_range, paste_instead_of_dbg);
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
    fn test_remove_dbg_target() {
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

    #[test]
    fn test_remove_dbg_keep_semicolon() {
        // https://github.com/rust-analyzer/rust-analyzer/issues/5129#issuecomment-651399779
        // not quite though
        // adding a comment at the end of the line makes
        // the ast::MacroCall to include the semicolon at the end
        let code = "
let res = <|>dbg!(1 * 20); // needless comment
";
        let expected = "
let res = 1 * 20; // needless comment
";
        check_assist(remove_dbg, code, expected);
    }

    #[test]
    fn test_remove_dbg_keep_expression() {
        let code = "
let res = <|>dbg!(a + b).foo();";
        let expected = "let res = (a + b).foo();";
        check_assist(remove_dbg, code, expected);
    }

    #[test]
    fn test_remove_dbg_from_inside_fn() {
        let code = "
fn square(x: u32) -> u32 {
    x * x
}

fn main() {
    let x = square(dbg<|>!(5 + 10));
    println!(\"{}\", x);
}";

        let expected = "
fn square(x: u32) -> u32 {
    x * x
}

fn main() {
    let x = square(5 + 10);
    println!(\"{}\", x);
}";
        check_assist_target(remove_dbg, code, "dbg!(5 + 10)");
        check_assist(remove_dbg, code, expected);
    }
}
