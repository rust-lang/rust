use hir::ModuleDef;
use ide_db::helpers::mod_path_to_ast;
use ide_db::items_locator;
use itertools::Itertools;
use syntax::{
    ast::{self, make, AstNode, NameOwner},
    SyntaxKind::{IDENT, WHITESPACE},
    TextSize,
};

use crate::{
    assist_context::{AssistBuilder, AssistContext, Assists},
    utils::{
        add_trait_assoc_items_to_impl, filter_assoc_items, generate_trait_impl_text,
        render_snippet, Cursor, DefaultMethods,
    },
    AssistId, AssistKind,
};

// Assist: replace_derive_with_manual_impl
//
// Converts a `derive` impl into a manual one.
//
// ```
// # trait Debug { fn fmt(&self, f: &mut Formatter) -> Result<()>; }
// #[derive(Deb$0ug, Display)]
// struct S;
// ```
// ->
// ```
// # trait Debug { fn fmt(&self, f: &mut Formatter) -> Result<()>; }
// #[derive(Display)]
// struct S;
//
// impl Debug for S {
//     fn fmt(&self, f: &mut Formatter) -> Result<()> {
//         ${0:todo!()}
//     }
// }
// ```
pub(crate) fn replace_derive_with_manual_impl(
    acc: &mut Assists,
    ctx: &AssistContext,
) -> Option<()> {
    let attr = ctx.find_node_at_offset::<ast::Attr>()?;

    let has_derive = attr
        .syntax()
        .descendants_with_tokens()
        .filter(|t| t.kind() == IDENT)
        .find_map(syntax::NodeOrToken::into_token)
        .filter(|t| t.text() == "derive")
        .is_some();
    if !has_derive {
        return None;
    }

    let trait_token = ctx.token_at_offset().find(|t| t.kind() == IDENT && t.text() != "derive")?;
    let trait_path = make::path_unqualified(make::path_segment(make::name_ref(trait_token.text())));

    let adt = attr.syntax().parent().and_then(ast::Adt::cast)?;
    let annotated_name = adt.name()?;
    let insert_pos = adt.syntax().text_range().end();

    let current_module = ctx.sema.scope(annotated_name.syntax()).module()?;
    let current_crate = current_module.krate();

    let found_traits = items_locator::with_for_exact_name(
        &ctx.sema,
        current_crate,
        trait_token.text().to_string(),
    )
    .into_iter()
    .filter_map(|item| match ModuleDef::from(item.as_module_def_id()?) {
        ModuleDef::Trait(trait_) => Some(trait_),
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
    for (trait_path, trait_) in found_traits.inspect(|_| no_traits_found = false) {
        add_assist(acc, ctx, &attr, &trait_path, Some(trait_), &adt, &annotated_name, insert_pos)?;
    }
    if no_traits_found {
        add_assist(acc, ctx, &attr, &trait_path, None, &adt, &annotated_name, insert_pos)?;
    }
    Some(())
}

fn add_assist(
    acc: &mut Assists,
    ctx: &AssistContext,
    attr: &ast::Attr,
    trait_path: &ast::Path,
    trait_: Option<hir::Trait>,
    adt: &ast::Adt,
    annotated_name: &ast::Name,
    insert_pos: TextSize,
) -> Option<()> {
    let target = attr.syntax().text_range();
    let input = attr.token_tree()?;
    let label = format!("Convert to manual  `impl {} for {}`", trait_path, annotated_name);
    let trait_name = trait_path.segment().and_then(|seg| seg.name_ref())?;

    acc.add(
        AssistId("replace_derive_with_manual_impl", AssistKind::Refactor),
        label,
        target,
        |builder| {
            let impl_def_with_items =
                impl_def_from_trait(&ctx.sema, annotated_name, trait_, trait_path);
            update_attribute(builder, &input, &trait_name, &attr);
            let trait_path = format!("{}", trait_path);
            match (ctx.config.snippet_cap, impl_def_with_items) {
                (None, _) => {
                    builder.insert(insert_pos, generate_trait_impl_text(adt, &trait_path, ""))
                }
                (Some(cap), None) => builder.insert_snippet(
                    cap,
                    insert_pos,
                    generate_trait_impl_text(adt, &trait_path, "    $0"),
                ),
                (Some(cap), Some((impl_def, first_assoc_item))) => {
                    let mut cursor = Cursor::Before(first_assoc_item.syntax());
                    let placeholder;
                    if let ast::AssocItem::Fn(ref func) = first_assoc_item {
                        if let Some(m) = func.syntax().descendants().find_map(ast::MacroCall::cast)
                        {
                            if m.syntax().text() == "todo!()" {
                                placeholder = m;
                                cursor = Cursor::Replace(placeholder.syntax());
                            }
                        }
                    }

                    builder.insert_snippet(
                        cap,
                        insert_pos,
                        format!("\n\n{}", render_snippet(cap, impl_def.syntax(), cursor)),
                    )
                }
            };
        },
    )
}

fn impl_def_from_trait(
    sema: &hir::Semantics<ide_db::RootDatabase>,
    annotated_name: &ast::Name,
    trait_: Option<hir::Trait>,
    trait_path: &ast::Path,
) -> Option<(ast::Impl, ast::AssocItem)> {
    let trait_ = trait_?;
    let target_scope = sema.scope(annotated_name.syntax());
    let trait_items = filter_assoc_items(sema.db, &trait_.items(sema.db), DefaultMethods::No);
    if trait_items.is_empty() {
        return None;
    }
    let impl_def = make::impl_trait(
        trait_path.clone(),
        make::path_unqualified(make::path_segment(make::name_ref(annotated_name.text()))),
    );
    let (impl_def, first_assoc_item) =
        add_trait_assoc_items_to_impl(sema, trait_items, trait_, impl_def, target_scope);
    Some((impl_def, first_assoc_item))
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
        .filter_map(|t| t.into_token().map(|t| t.text().to_string()))
        .filter(|t| t != trait_name.text())
        .collect::<Vec<_>>();
    let has_more_derives = !new_attr_input.is_empty();

    if has_more_derives {
        let new_attr_input = format!("({})", new_attr_input.iter().format(", "));
        builder.replace(input.syntax().text_range(), new_attr_input);
    } else {
        let attr_range = attr.syntax().text_range();
        builder.delete(attr_range);

        if let Some(line_break_range) = attr
            .syntax()
            .next_sibling_or_token()
            .filter(|t| t.kind() == WHITESPACE)
            .map(|t| t.text_range())
        {
            builder.delete(line_break_range);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn add_custom_impl_debug() {
        check_assist(
            replace_derive_with_manual_impl,
            "
mod fmt {
    pub struct Error;
    pub type Result = Result<(), Error>;
    pub struct Formatter<'a>;
    pub trait Debug {
        fn fmt(&self, f: &mut Formatter<'_>) -> Result;
    }
}

#[derive(Debu$0g)]
struct Foo {
    bar: String,
}
",
            "
mod fmt {
    pub struct Error;
    pub type Result = Result<(), Error>;
    pub struct Formatter<'a>;
    pub trait Debug {
        fn fmt(&self, f: &mut Formatter<'_>) -> Result;
    }
}

struct Foo {
    bar: String,
}

impl fmt::Debug for Foo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        ${0:todo!()}
    }
}
",
        )
    }
    #[test]
    fn add_custom_impl_all() {
        check_assist(
            replace_derive_with_manual_impl,
            "
mod foo {
    pub trait Bar {
        type Qux;
        const Baz: usize = 42;
        const Fez: usize;
        fn foo();
        fn bar() {}
    }
}

#[derive($0Bar)]
struct Foo {
    bar: String,
}
",
            "
mod foo {
    pub trait Bar {
        type Qux;
        const Baz: usize = 42;
        const Fez: usize;
        fn foo();
        fn bar() {}
    }
}

struct Foo {
    bar: String,
}

impl foo::Bar for Foo {
    $0type Qux;

    const Baz: usize = 42;

    const Fez: usize;

    fn foo() {
        todo!()
    }
}
",
        )
    }
    #[test]
    fn add_custom_impl_for_unique_input() {
        check_assist(
            replace_derive_with_manual_impl,
            "
#[derive(Debu$0g)]
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
            replace_derive_with_manual_impl,
            "
#[derive(Debug$0)]
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
            replace_derive_with_manual_impl,
            "
#[derive(Display, Debug$0, Serialize)]
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
            replace_derive_with_manual_impl,
            "
#[derive($0)]
struct Foo {}
            ",
        )
    }

    #[test]
    fn test_ignore_if_cursor_on_param() {
        check_assist_not_applicable(
            replace_derive_with_manual_impl,
            "
#[derive$0(Debug)]
struct Foo {}
            ",
        );

        check_assist_not_applicable(
            replace_derive_with_manual_impl,
            "
#[derive(Debug)$0]
struct Foo {}
            ",
        )
    }

    #[test]
    fn test_ignore_if_not_derive() {
        check_assist_not_applicable(
            replace_derive_with_manual_impl,
            "
#[allow(non_camel_$0case_types)]
struct Foo {}
            ",
        )
    }
}
