//! FIXME: write short doc here

use crate::{Assist, AssistCtx, AssistId};
use hir::db::HirDatabase;
use join_to_string::join;
use ra_syntax::{
    ast::{self, AstNode},
    Direction, SmolStr,
    SyntaxKind::{IDENT, WHITESPACE},
    TextRange, TextUnit,
};

const DERIVE_TRAIT: &'static str = "derive";

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
//
// }
// ```
pub(crate) fn add_custom_impl(ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let input = ctx.find_node_at_offset::<ast::AttrInput>()?;
    let attr = input.syntax().parent().and_then(ast::Attr::cast)?;

    let attr_name = attr
        .syntax()
        .descendants_with_tokens()
        .filter(|t| t.kind() == IDENT)
        .find_map(|i| i.into_token())
        .filter(|t| *t.text() == DERIVE_TRAIT)?
        .text()
        .clone();

    let trait_token =
        ctx.token_at_offset().filter(|t| t.kind() == IDENT && *t.text() != attr_name).next()?;

    let annotated = attr.syntax().siblings(Direction::Next).find_map(|s| ast::Name::cast(s))?;
    let annotated_name = annotated.syntax().text().to_string();
    let start_offset = annotated.syntax().parent()?.text_range().end();

    ctx.add_assist(AssistId("add_custom_impl"), "add custom impl", |edit| {
        edit.target(attr.syntax().text_range());

        let new_attr_input = input
            .syntax()
            .descendants_with_tokens()
            .filter(|t| t.kind() == IDENT)
            .filter_map(|t| t.into_token().map(|t| t.text().clone()))
            .filter(|t| t != trait_token.text())
            .collect::<Vec<SmolStr>>();
        let has_more_derives = new_attr_input.len() > 0;
        let new_attr_input =
            join(new_attr_input.iter()).separator(", ").surround_with("(", ")").to_string();
        let new_attr_input_len = new_attr_input.len();

        let mut buf = String::new();
        buf.push_str("\n\nimpl ");
        buf.push_str(trait_token.text().as_str());
        buf.push_str(" for ");
        buf.push_str(annotated_name.as_str());
        buf.push_str(" {\n");

        let cursor_delta = if has_more_derives {
            edit.replace(input.syntax().text_range(), new_attr_input);
            input.syntax().text_range().len() - TextUnit::from_usize(new_attr_input_len)
        } else {
            let attr_range = attr.syntax().text_range();
            edit.delete(attr_range);

            let line_break_range = attr
                .syntax()
                .next_sibling_or_token()
                .filter(|t| t.kind() == WHITESPACE)
                .map(|t| t.text_range())
                .unwrap_or(TextRange::from_to(TextUnit::from(0), TextUnit::from(0)));
            edit.delete(line_break_range);

            attr_range.len() + line_break_range.len()
        };

        edit.set_cursor(start_offset + TextUnit::of_str(&buf) - cursor_delta);
        buf.push_str("\n}");
        edit.insert(start_offset, buf);
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::{check_assist, check_assist_not_applicable};

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
<|>
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
<|>
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
<|>
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
