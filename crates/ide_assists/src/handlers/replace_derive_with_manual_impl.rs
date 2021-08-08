use hir::ModuleDef;
use ide_db::helpers::{import_assets::NameToImport, mod_path_to_ast};
use ide_db::items_locator;
use itertools::Itertools;
use syntax::ast::edit::AstNodeEdit;
use syntax::ast::Pat;
use syntax::ted;
use syntax::{
    ast::{self, make, AstNode, NameOwner},
    SyntaxKind::{IDENT, WHITESPACE},
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
//     $0fn fmt(&self, f: &mut Formatter) -> Result<()> {
//         f.debug_struct("S").finish()
//     }
// }
// ```
pub(crate) fn replace_derive_with_manual_impl(
    acc: &mut Assists,
    ctx: &AssistContext,
) -> Option<()> {
    let attr = ctx.find_node_at_offset::<ast::Attr>()?;
    let (name, args) = attr.as_simple_call()?;
    if name != "derive" {
        return None;
    }

    if !args.syntax().text_range().contains(ctx.offset()) {
        cov_mark::hit!(outside_of_attr_args);
        return None;
    }

    let trait_token = args.syntax().token_at_offset(ctx.offset()).find(|t| t.kind() == IDENT)?;
    let trait_name = trait_token.text();

    let adt = attr.syntax().parent().and_then(ast::Adt::cast)?;

    let current_module = ctx.sema.scope(adt.syntax()).module()?;
    let current_crate = current_module.krate();

    let found_traits = items_locator::items_with_name(
        &ctx.sema,
        current_crate,
        NameToImport::Exact(trait_name.to_string()),
        items_locator::AssocItemSearch::Exclude,
        Some(items_locator::DEFAULT_QUERY_SEARCH_LIMIT.inner()),
    )
    .filter_map(|item| match item.as_module_def()? {
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
        add_assist(acc, ctx, &attr, &args, &trait_path, Some(trait_), &adt)?;
    }
    if no_traits_found {
        let trait_path = make::ext::ident_path(trait_name);
        add_assist(acc, ctx, &attr, &args, &trait_path, None, &adt)?;
    }
    Some(())
}

fn add_assist(
    acc: &mut Assists,
    ctx: &AssistContext,
    attr: &ast::Attr,
    input: &ast::TokenTree,
    trait_path: &ast::Path,
    trait_: Option<hir::Trait>,
    adt: &ast::Adt,
) -> Option<()> {
    let target = attr.syntax().text_range();
    let annotated_name = adt.name()?;
    let label = format!("Convert to manual `impl {} for {}`", trait_path, annotated_name);
    let trait_name = trait_path.segment().and_then(|seg| seg.name_ref())?;

    acc.add(
        AssistId("replace_derive_with_manual_impl", AssistKind::Refactor),
        label,
        target,
        |builder| {
            let insert_pos = adt.syntax().text_range().end();
            let impl_def_with_items =
                impl_def_from_trait(&ctx.sema, adt, &annotated_name, trait_, trait_path);
            update_attribute(builder, input, &trait_name, attr);
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
                        // need to know what kind of derive this is: if it's Derive Debug, special case it.
                        // the name of the struct
                        // list of fields of the struct
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
    adt: &ast::Adt,
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
    let impl_def =
        make::impl_trait(trait_path.clone(), make::ext::ident_path(&annotated_name.text()));
    let (impl_def, first_assoc_item) =
        add_trait_assoc_items_to_impl(sema, trait_items, trait_, impl_def, target_scope);

    if let ast::AssocItem::Fn(fn_) = &first_assoc_item {
        if trait_path.segment().unwrap().name_ref().unwrap().text() == "Debug" {
            gen_debug_impl(adt, fn_, annotated_name);
        }
    }
    Some((impl_def, first_assoc_item))
}

fn gen_debug_impl(adt: &ast::Adt, fn_: &ast::Fn, annotated_name: &ast::Name) {
    match adt {
        ast::Adt::Union(_) => {} // `Debug` cannot be derived for unions, so no default impl can be provided.
        ast::Adt::Enum(_) => {}  // TODO
        ast::Adt::Struct(strukt) => match strukt.field_list() {
            Some(ast::FieldList::RecordFieldList(field_list)) => {
                let name = format!("\"{}\"", annotated_name);
                let args = make::arg_list(Some(make::expr_literal(&name).into()));
                let target = make::expr_path(make::ext::ident_path("f"));
                let mut expr = make::expr_method_call(target, "debug_struct", args);
                for field in field_list.fields() {
                    if let Some(name) = field.name() {
                        let f_name = make::expr_literal(&(format!("\"{}\"", name))).into();
                        let f_path = make::expr_path(make::ext::ident_path("self"));
                        let f_path = make::expr_ref(f_path, false);
                        let f_path = make::expr_field(f_path, &format!("{}", name)).into();
                        let args = make::arg_list(vec![f_name, f_path]);
                        expr = make::expr_method_call(expr, "field", args);
                    }
                }
                let expr = make::expr_method_call(expr, "finish", make::arg_list(None));
                let body = make::block_expr(None, Some(expr)).indent(ast::edit::IndentLevel(1));
                ted::replace(fn_.body().unwrap().syntax(), body.clone_for_update().syntax());
            }
            Some(ast::FieldList::TupleFieldList(field_list)) => {}
            None => {
                let name = format!("\"{}\"", annotated_name);
                let args = make::arg_list(Some(make::expr_literal(&name).into()));
                let target = make::expr_path(make::ext::ident_path("f"));
                let expr = make::expr_method_call(target, "debug_struct", args);
                let expr = make::expr_method_call(expr, "finish", make::arg_list(None));
                let body = make::block_expr(None, Some(expr)).indent(ast::edit::IndentLevel(1));
                ted::replace(fn_.body().unwrap().syntax(), body.clone_for_update().syntax());
            }
        },
    }
}

fn update_attribute(
    builder: &mut AssistBuilder,
    input: &ast::TokenTree,
    trait_name: &ast::NameRef,
    attr: &ast::Attr,
) {
    let trait_name = trait_name.text();
    let new_attr_input = input
        .syntax()
        .descendants_with_tokens()
        .filter(|t| t.kind() == IDENT)
        .filter_map(|t| t.into_token().map(|t| t.text().to_string()))
        .filter(|t| t != &trait_name)
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
            r#"
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
"#,
            r#"
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
    $0fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Foo").field("bar", &self.bar).finish()
    }
}
"#,
        )
    }
    #[test]
    fn add_custom_impl_all() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
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
"#,
            r#"
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
"#,
        )
    }
    #[test]
    fn add_custom_impl_for_unique_input() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
#[derive(Debu$0g)]
struct Foo {
    bar: String,
}
            "#,
            r#"
struct Foo {
    bar: String,
}

impl Debug for Foo {
    $0
}
            "#,
        )
    }

    #[test]
    fn add_custom_impl_for_with_visibility_modifier() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
#[derive(Debug$0)]
pub struct Foo {
    bar: String,
}
            "#,
            r#"
pub struct Foo {
    bar: String,
}

impl Debug for Foo {
    $0
}
            "#,
        )
    }

    #[test]
    fn add_custom_impl_when_multiple_inputs() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
#[derive(Display, Debug$0, Serialize)]
struct Foo {}
            "#,
            r#"
#[derive(Display, Serialize)]
struct Foo {}

impl Debug for Foo {
    $0
}
            "#,
        )
    }

    #[test]
    fn test_ignore_derive_macro_without_input() {
        check_assist_not_applicable(
            replace_derive_with_manual_impl,
            r#"
#[derive($0)]
struct Foo {}
            "#,
        )
    }

    #[test]
    fn test_ignore_if_cursor_on_param() {
        check_assist_not_applicable(
            replace_derive_with_manual_impl,
            r#"
#[derive$0(Debug)]
struct Foo {}
            "#,
        );

        check_assist_not_applicable(
            replace_derive_with_manual_impl,
            r#"
#[derive(Debug)$0]
struct Foo {}
            "#,
        )
    }

    #[test]
    fn test_ignore_if_not_derive() {
        check_assist_not_applicable(
            replace_derive_with_manual_impl,
            r#"
#[allow(non_camel_$0case_types)]
struct Foo {}
            "#,
        )
    }

    #[test]
    fn works_at_start_of_file() {
        cov_mark::check!(outside_of_attr_args);
        check_assist_not_applicable(
            replace_derive_with_manual_impl,
            r#"
$0#[derive(Debug)]
struct S;
            "#,
        );
    }
}
