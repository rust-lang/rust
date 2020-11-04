use ide_db::imports_locator;
use itertools::Itertools;
use syntax::{
    ast::{self, make, AstNode},
    Direction, SmolStr,
    SyntaxKind::{IDENT, WHITESPACE},
    TextRange, TextSize,
};

use crate::{
    assist_config::SnippetCap,
    assist_context::{AssistBuilder, AssistContext, Assists},
    utils::mod_path_to_ast,
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
    let attr = ctx.find_node_at_offset::<ast::Attr>()?;

    let attr_name = attr
        .syntax()
        .descendants_with_tokens()
        .filter(|t| t.kind() == IDENT)
        .find_map(syntax::NodeOrToken::into_token)
        .filter(|t| t.text() == "derive")?
        .text()
        .clone();

    let trait_token =
        ctx.token_at_offset().find(|t| t.kind() == IDENT && *t.text() != attr_name)?;
    let trait_path = make::path_unqualified(make::path_segment(make::name_ref(trait_token.text())));

    let annotated = attr.syntax().siblings(Direction::Next).find_map(ast::Name::cast)?;
    let annotated_name = annotated.syntax().text().to_string();
    let insert_pos = annotated.syntax().parent()?.text_range().end();

    let current_module = ctx.sema.scope(annotated.syntax()).module()?;
    let current_crate = current_module.krate();

    let found_traits = imports_locator::find_imports(&ctx.sema, current_crate, trait_token.text())
        .into_iter()
        .filter_map(|candidate: either::Either<hir::ModuleDef, hir::MacroDef>| match candidate {
            either::Either::Left(hir::ModuleDef::Trait(trait_)) => Some(trait_),
            _ => None,
        })
        .flat_map(|trait_| {
            current_module
                .find_use_path(ctx.sema.db, hir::ModuleDef::Trait(trait_))
                .as_ref()
                .map(mod_path_to_ast)
                .zip(Some(trait_))
        });

    let mut no_traits_found = true;
    for (trait_path, _trait) in found_traits.inspect(|_| no_traits_found = false) {
        add_assist(acc, ctx.config.snippet_cap, &attr, &trait_path, &annotated_name, insert_pos)?;
    }
    if no_traits_found {
        add_assist(acc, ctx.config.snippet_cap, &attr, &trait_path, &annotated_name, insert_pos)?;
    }
    Some(())
}

fn add_assist(
    acc: &mut Assists,
    snippet_cap: Option<SnippetCap>,
    attr: &ast::Attr,
    trait_path: &ast::Path,
    annotated_name: &str,
    insert_pos: TextSize,
) -> Option<()> {
    let target = attr.syntax().text_range();
    let input = attr.token_tree()?;
    let label = format!("Add custom impl `{}` for `{}`", trait_path, annotated_name);
    let trait_name = trait_path.segment().and_then(|seg| seg.name_ref())?;

    acc.add(AssistId("add_custom_impl", AssistKind::Refactor), label, target, |builder| {
        update_attribute(builder, &input, &trait_name, &attr);
        match snippet_cap {
            Some(cap) => {
                builder.insert_snippet(
                    cap,
                    insert_pos,
                    format!("\n\nimpl {} for {} {{\n    $0\n}}", trait_path, annotated_name),
                );
            }
            None => {
                builder.insert(
                    insert_pos,
                    format!("\n\nimpl {} for {} {{\n\n}}", trait_path, annotated_name),
                );
            }
        }
    })
}

fn update_attribute(
    builder: &mut AssistBuilder,
    input: &ast::TokenTree,
    trait_name: &ast::NameRef,
    attr: &ast::Attr,
) {
    let new_attr_input = input
        .syntax()
        .descendants_with_tokens()
        .filter(|t| t.kind() == IDENT)
        .filter_map(|t| t.into_token().map(|t| t.text().clone()))
        .filter(|t| t != trait_name.text())
        .collect::<Vec<SmolStr>>();
    let has_more_derives = !new_attr_input.is_empty();

    if has_more_derives {
        let new_attr_input = format!("({})", new_attr_input.iter().format(", "));
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
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn add_custom_impl_qualified() {
        check_assist(
            add_custom_impl,
            "
mod fmt {
    pub trait Debug {}
}

#[derive(Debu<|>g)]
struct Foo {
    bar: String,
}
",
            "
mod fmt {
    pub trait Debug {}
}

struct Foo {
    bar: String,
}

impl fmt::Debug for Foo {
    $0
}
",
        )
    }
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
