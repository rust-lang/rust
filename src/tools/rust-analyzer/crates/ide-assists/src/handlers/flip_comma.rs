use syntax::{
    AstNode, Direction, NodeOrToken, SyntaxKind, SyntaxToken, T,
    algo::non_trivia_sibling,
    ast::{self, syntax_factory::SyntaxFactory},
    syntax_editor::SyntaxMapping,
};

use crate::{AssistContext, AssistId, Assists};

// Assist: flip_comma
//
// Flips two comma-separated items.
//
// ```
// fn main() {
//     ((1, 2),$0 (3, 4));
// }
// ```
// ->
// ```
// fn main() {
//     ((3, 4), (1, 2));
// }
// ```
pub(crate) fn flip_comma(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let comma = ctx.find_token_syntax_at_offset(T![,])?;
    let prev = non_trivia_sibling(comma.clone().into(), Direction::Prev)?;
    let next = non_trivia_sibling(comma.clone().into(), Direction::Next)?;

    // Don't apply a "flip" in case of a last comma
    // that typically comes before punctuation
    if next.kind().is_punct() {
        return None;
    }

    // Don't apply a "flip" inside the macro call
    // since macro input are just mere tokens
    if comma.parent_ancestors().any(|it| it.kind() == SyntaxKind::MACRO_CALL) {
        return None;
    }

    let target = comma.text_range();
    acc.add(AssistId::refactor_rewrite("flip_comma"), "Flip comma", target, |builder| {
        let parent = comma.parent().unwrap();
        let mut editor = builder.make_editor(&parent);

        if let Some(parent) = ast::TokenTree::cast(parent) {
            // An attribute. It often contains a path followed by a
            // token tree (e.g. `align(2)`), so we have to be smarter.
            let (new_tree, mapping) = flip_tree(parent.clone(), comma);
            editor.replace(parent.syntax(), new_tree.syntax());
            editor.add_mappings(mapping);
        } else {
            editor.replace(prev.clone(), next.clone());
            editor.replace(next.clone(), prev.clone());
        }

        builder.add_file_edits(ctx.vfs_file_id(), editor);
    })
}

fn flip_tree(tree: ast::TokenTree, comma: SyntaxToken) -> (ast::TokenTree, SyntaxMapping) {
    let mut tree_iter = tree.token_trees_and_tokens();
    let before: Vec<_> =
        tree_iter.by_ref().take_while(|it| it.as_token() != Some(&comma)).collect();
    let after: Vec<_> = tree_iter.collect();

    let not_ws = |element: &NodeOrToken<_, SyntaxToken>| match element {
        NodeOrToken::Token(token) => token.kind() != SyntaxKind::WHITESPACE,
        NodeOrToken::Node(_) => true,
    };

    let is_comma = |element: &NodeOrToken<_, SyntaxToken>| match element {
        NodeOrToken::Token(token) => token.kind() == T![,],
        NodeOrToken::Node(_) => false,
    };

    let prev_start_untrimmed = match before.iter().rposition(is_comma) {
        Some(pos) => pos + 1,
        None => 1,
    };
    let prev_end = 1 + before.iter().rposition(not_ws).unwrap();
    let prev_start = prev_start_untrimmed
        + before[prev_start_untrimmed..prev_end].iter().position(not_ws).unwrap();

    let next_start = after.iter().position(not_ws).unwrap();
    let next_end_untrimmed = match after.iter().position(is_comma) {
        Some(pos) => pos,
        None => after.len() - 1,
    };
    let next_end = 1 + after[..next_end_untrimmed].iter().rposition(not_ws).unwrap();

    let result = [
        &before[1..prev_start],
        &after[next_start..next_end],
        &before[prev_end..],
        &[NodeOrToken::Token(comma)],
        &after[..next_start],
        &before[prev_start..prev_end],
        &after[next_end..after.len() - 1],
    ]
    .concat();

    let make = SyntaxFactory::with_mappings();
    let new_token_tree = make.token_tree(tree.left_delimiter_token().unwrap().kind(), result);
    (new_token_tree, make.finish_with_mappings())
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::{check_assist, check_assist_not_applicable, check_assist_target};

    #[test]
    fn flip_comma_works_for_function_parameters() {
        check_assist(
            flip_comma,
            r#"fn foo(x: i32,$0 y: Result<(), ()>) {}"#,
            r#"fn foo(y: Result<(), ()>, x: i32) {}"#,
        )
    }

    #[test]
    fn flip_comma_target() {
        check_assist_target(flip_comma, r#"fn foo(x: i32,$0 y: Result<(), ()>) {}"#, ",")
    }

    #[test]
    fn flip_comma_before_punct() {
        // See https://github.com/rust-lang/rust-analyzer/issues/1619
        // "Flip comma" assist shouldn't be applicable to the last comma in enum or struct
        // declaration body.
        check_assist_not_applicable(flip_comma, "pub enum Test { A,$0 }");
        check_assist_not_applicable(flip_comma, "pub struct Test { foo: usize,$0 }");
    }

    #[test]
    fn flip_comma_works() {
        check_assist(
            flip_comma,
            r#"fn main() {((1, 2),$0 (3, 4));}"#,
            r#"fn main() {((3, 4), (1, 2));}"#,
        )
    }

    #[test]
    fn flip_comma_not_applicable_for_macro_input() {
        // "Flip comma" assist shouldn't be applicable inside the macro call
        // See https://github.com/rust-lang/rust-analyzer/issues/7693
        check_assist_not_applicable(flip_comma, r#"bar!(a,$0 b)"#);
    }

    #[test]
    fn flip_comma_attribute() {
        check_assist(
            flip_comma,
            r#"#[repr(align(2),$0 C)] struct Foo;"#,
            r#"#[repr(C, align(2))] struct Foo;"#,
        );
        check_assist(
            flip_comma,
            r#"#[foo(bar, baz(1 + 1),$0 qux, other)] struct Foo;"#,
            r#"#[foo(bar, qux, baz(1 + 1), other)] struct Foo;"#,
        );
    }

    #[test]
    fn flip_comma_attribute_incomplete() {
        check_assist_not_applicable(flip_comma, r#"#[repr(align(2),$0)] struct Foo;"#);
    }
}
