use ra_syntax::{
    ast::{self, AstNode},
    Direction, SmolStr,
    SyntaxKind::{IDENT, WHITESPACE},
    TextRange, TextSize,
};
use stdx::SepBy;

use crate::{
    assist_context::{AssistContext, Assists},
    AssistId, AssistKind,
};

// Assist: add_custom_impl
//
// Adds impl block for derived trait.
//
// ```
// #[derive(Deb<|>ug, Display)]
// struct S;
// ```
// ->
// ```
// #[derive(Display)]
// struct S;
//
// impl Debug for S {
//     $0
// }
// ```
pub(crate) fn add_custom_impl(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let input = ctx.find_node_at_offset::<ast::AttrInput>()?;
    let attr = input.syntax().parent().and_then(ast::Attr::cast)?;

    let attr_name = attr
        .syntax()
        .descendants_with_tokens()
        .filter(|t| t.kind() == IDENT)
        .find_map(|i| i.into_token())
        .filter(|t| *t.text() == "derive")?
        .text()
        .clone();

    let trait_token =
        ctx.token_at_offset().find(|t| t.kind() == IDENT && *t.text() != attr_name)?;

    let annotated = attr.syntax().siblings(Direction::Next).find_map(ast::Name::cast)?;
    let annotated_name = annotated.syntax().text().to_string();
    let start_offset = annotated.syntax().parent()?.text_range().end();

    let label =
        format!("Add custom impl `{}` for `{}`", trait_token.text().as_str(), annotated_name);

    let target = attr.syntax().text_range();
    acc.add(AssistId("add_custom_impl", AssistKind::Refactor), label, target, |builder| {
        let new_attr_input = input
            .syntax()
            .descendants_with_tokens()
            .filter(|t| t.kind() == IDENT)
            .filter_map(|t| t.into_token().map(|t| t.text().clone()))
            .filter(|t| t != trait_token.text())
            .collect::<Vec<SmolStr>>();
        let has_more_derives = !new_attr_input.is_empty();
        let new_attr_input = new_attr_input.iter().sep_by(", ").surround_with("(", ")").to_string();

        if has_more_derives {
            builder.replace(input.syntax().text_range(), new_attr_input);
        } else {
            let attr_range = attr.syntax().text_range();
            builder.delete(attr_range);

            let line_break_range = attr
                .syntax()
                .next_sibling_or_token()
                .filter(|t| t.kind() == WHITESPACE)
                .map(|t| t.text_range())
                .unwrap_or_else(|| TextRange::new(TextSize::from(0), TextSize::from(0)));
            builder.delete(line_break_range);
        }

        match ctx.config.snippet_cap {
            Some(cap) => {
                builder.insert_snippet(
                    cap,
                    start_offset,
                    format!("\n\nimpl {} for {} {{\n    $0\n}}", trait_token, annotated_name),
                );
            }
            None => {
                builder.insert(
                    start_offset,
                    format!("\n\nimpl {} for {} {{\n\n}}", trait_token, annotated_name),
                );
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn add_custom_impl_for_unique_input() {
        check_assist(
            add_custom_impl,
            "
#[derive(Debu<|>g)]
struct Foo {
    bar: String,
}
            ",
            "
struct Foo {
    bar: String,
}

impl Debug for Foo {
    $0
}
            ",
        )
    }

    #[test]
    fn add_custom_impl_for_with_visibility_modifier() {
        check_assist(
            add_custom_impl,
            "
#[derive(Debug<|>)]
pub struct Foo {
    bar: String,
}
            ",
            "
pub struct Foo {
    bar: String,
}

impl Debug for Foo {
    $0
}
            ",
        )
    }

    #[test]
    fn add_custom_impl_when_multiple_inputs() {
        check_assist(
            add_custom_impl,
            "
#[derive(Display, Debug<|>, Serialize)]
struct Foo {}
            ",
            "
#[derive(Display, Serialize)]
struct Foo {}

impl Debug for Foo {
    $0
}
            ",
        )
    }

    #[test]
    fn test_ignore_derive_macro_without_input() {
        check_assist_not_applicable(
            add_custom_impl,
            "
#[derive(<|>)]
struct Foo {}
            ",
        )
    }

    #[test]
    fn test_ignore_if_cursor_on_param() {
        check_assist_not_applicable(
            add_custom_impl,
            "
#[derive<|>(Debug)]
struct Foo {}
            ",
        );

        check_assist_not_applicable(
            add_custom_impl,
            "
#[derive(Debug)<|>]
struct Foo {}
            ",
        )
    }

    #[test]
    fn test_ignore_if_not_derive() {
        check_assist_not_applicable(
            add_custom_impl,
            "
#[allow(non_camel_<|>case_types)]
struct Foo {}
            ",
        )
    }
}
