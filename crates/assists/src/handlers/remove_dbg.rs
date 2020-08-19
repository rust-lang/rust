use syntax::{
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

    let is_leaf = macro_call.syntax().next_sibling().is_none();

    let macro_end = if macro_call.semicolon_token().is_some() {
        macro_call.syntax().text_range().end() - TextSize::of(';')
    } else {
        macro_call.syntax().text_range().end()
    };

    // macro_range determines what will be deleted and replaced with macro_content
    let macro_range = TextRange::new(macro_call.syntax().text_range().start(), macro_end);
    let paste_instead_of_dbg = {
        let text = macro_call.token_tree()?.syntax().text();

        // leafiness determines if we should include the parenthesis or not
        let slice_index: TextRange = if is_leaf {
            // leaf means - we can extract the contents of the dbg! in text
            TextRange::new(TextSize::of('('), text.len() - TextSize::of(')'))
        } else {
            // not leaf - means we should keep the parens
            TextRange::up_to(text.len())
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
    use crate::tests::{check_assist, check_assist_not_applicable, check_assist_target};

    use super::*;

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
        check_assist(
            remove_dbg,
            r#"let res = <|>dbg!(1 * 20); // needless comment"#,
            r#"let res = 1 * 20; // needless comment"#,
        );
    }

    #[test]
    fn test_remove_dbg_keep_expression() {
        check_assist(
            remove_dbg,
            r#"let res = <|>dbg!(a + b).foo();"#,
            r#"let res = (a + b).foo();"#,
        );
    }

    #[test]
    fn test_remove_dbg_from_inside_fn() {
        check_assist_target(
            remove_dbg,
            r#"
fn square(x: u32) -> u32 {
    x * x
}

fn main() {
    let x = square(dbg<|>!(5 + 10));
    println!("{}", x);
}"#,
            "dbg!(5 + 10)",
        );

        check_assist(
            remove_dbg,
            r#"
fn square(x: u32) -> u32 {
    x * x
}

fn main() {
    let x = square(dbg<|>!(5 + 10));
    println!("{}", x);
}"#,
            r#"
fn square(x: u32) -> u32 {
    x * x
}

fn main() {
    let x = square(5 + 10);
    println!("{}", x);
}"#,
        );
    }
}
