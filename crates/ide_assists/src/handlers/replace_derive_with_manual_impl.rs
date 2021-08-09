use hir::ModuleDef;
use ide_db::helpers::{import_assets::NameToImport, mod_path_to_ast};
use ide_db::items_locator;
use itertools::Itertools;
use syntax::ast::edit::AstNodeEdit;
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

    // Generate a default `impl` function body for the derived trait.
    if let ast::AssocItem::Fn(ref func) = first_assoc_item {
        let _ = gen_trait_body_impl(func, trait_path, adt, annotated_name);
    };

    Some((impl_def, first_assoc_item))
}

/// Generate custom trait bodies where possible.
///
/// Returns `Option` so that we can use `?` rather than `if let Some`. Returning
/// `None` means that generating a custom trait body failed, and the body will remain
/// as `todo!` instead.
fn gen_trait_body_impl(
    func: &ast::Fn,
    trait_path: &ast::Path,
    adt: &ast::Adt,
    annotated_name: &ast::Name,
) -> Option<()> {
    match trait_path.segment()?.name_ref()?.text().as_str() {
        "Debug" => gen_debug_impl(adt, func, annotated_name),
        "Default" => gen_default_impl(adt, func),
        _ => None,
    }
}

/// Generate a `Debug` impl based on the fields and members of the target type.
fn gen_debug_impl(adt: &ast::Adt, func: &ast::Fn, annotated_name: &ast::Name) -> Option<()> {
    match adt {
        // `Debug` cannot be derived for unions, so no default impl can be provided.
        ast::Adt::Union(_) => None,

        // => match self { Self::Variant => write!(f, "Variant") }
        ast::Adt::Enum(enum_) => {
            let list = enum_.variant_list()?;
            let mut arms = vec![];
            for variant in list.variants() {
                let name = variant.name()?;
                let left = make::ext::ident_path("Self");
                let right = make::ext::ident_path(&format!("{}", name));
                let variant_name = make::path_pat(make::path_concat(left, right));

                let target = make::expr_path(make::ext::ident_path("f").into());
                let fmt_string = make::expr_literal(&(format!("\"{}\"", name))).into();
                let args = make::arg_list(vec![target, fmt_string]);
                let macro_name = make::expr_path(make::ext::ident_path("write"));
                let macro_call = make::expr_macro_call(macro_name, args);

                arms.push(make::match_arm(Some(variant_name.into()), None, macro_call.into()));
            }

            let match_target = make::expr_path(make::ext::ident_path("self"));
            let list = make::match_arm_list(arms).indent(ast::edit::IndentLevel(1));
            let match_expr = make::expr_match(match_target, list);

            let body = make::block_expr(None, Some(match_expr));
            let body = body.indent(ast::edit::IndentLevel(1));
            ted::replace(func.body()?.syntax(), body.clone_for_update().syntax());
            Some(())
        }

        ast::Adt::Struct(strukt) => {
            let name = format!("\"{}\"", annotated_name);
            let args = make::arg_list(Some(make::expr_literal(&name).into()));
            let target = make::expr_path(make::ext::ident_path("f"));

            let expr = match strukt.field_list() {
                // => f.debug_struct("Name").finish()
                None => make::expr_method_call(target, make::name_ref("debug_struct"), args),

                // => f.debug_struct("Name").field("foo", &self.foo).finish()
                Some(ast::FieldList::RecordFieldList(field_list)) => {
                    let method = make::name_ref("debug_struct");
                    let mut expr = make::expr_method_call(target, method, args);
                    for field in field_list.fields() {
                        let name = field.name()?;
                        let f_name = make::expr_literal(&(format!("\"{}\"", name))).into();
                        let f_path = make::expr_path(make::ext::ident_path("self"));
                        let f_path = make::expr_ref(f_path, false);
                        let f_path = make::expr_field(f_path, &format!("{}", name)).into();
                        let args = make::arg_list(vec![f_name, f_path]);
                        expr = make::expr_method_call(expr, make::name_ref("field"), args);
                    }
                    expr
                }

                // => f.debug_tuple("Name").field(self.0).finish()
                Some(ast::FieldList::TupleFieldList(field_list)) => {
                    let method = make::name_ref("debug_tuple");
                    let mut expr = make::expr_method_call(target, method, args);
                    for (idx, _) in field_list.fields().enumerate() {
                        let f_path = make::expr_path(make::ext::ident_path("self"));
                        let f_path = make::expr_ref(f_path, false);
                        let f_path = make::expr_field(f_path, &format!("{}", idx)).into();
                        let method = make::name_ref("field");
                        expr = make::expr_method_call(expr, method, make::arg_list(Some(f_path)));
                    }
                    expr
                }
            };

            let method = make::name_ref("finish");
            let expr = make::expr_method_call(expr, method, make::arg_list(None));
            let body = make::block_expr(None, Some(expr)).indent(ast::edit::IndentLevel(1));
            ted::replace(func.body()?.syntax(), body.clone_for_update().syntax());
            Some(())
        }
    }
}

/// Generate a `Debug` impl based on the fields and members of the target type.
fn gen_default_impl(adt: &ast::Adt, func: &ast::Fn) -> Option<()> {
    fn gen_default_call() -> ast::Expr {
        let trait_name = make::ext::ident_path("Default");
        let method_name = make::ext::ident_path("default");
        let fn_name = make::expr_path(make::path_concat(trait_name, method_name));
        make::expr_call(fn_name, make::arg_list(None))
    }
    match adt {
        // `Debug` cannot be derived for unions, so no default impl can be provided.
        ast::Adt::Union(_) => None,
        // Deriving `Debug` for enums is not stable yet.
        ast::Adt::Enum(_) => None,
        ast::Adt::Struct(strukt) => {
            let expr = match strukt.field_list() {
                Some(ast::FieldList::RecordFieldList(field_list)) => {
                    let mut fields = vec![];
                    for field in field_list.fields() {
                        let method_call = gen_default_call();
                        let name_ref = make::name_ref(&field.name()?.to_string());
                        let field = make::record_expr_field(name_ref, Some(method_call));
                        fields.push(field);
                    }
                    let struct_name = make::ext::ident_path("Self");
                    let fields = make::record_expr_field_list(fields);
                    make::record_expr(struct_name, fields).into()
                }
                Some(ast::FieldList::TupleFieldList(field_list)) => {
                    let struct_name = make::expr_path(make::ext::ident_path("Self"));
                    let fields = field_list.fields().map(|_| gen_default_call());
                    make::expr_call(struct_name, make::arg_list(fields))
                }
                None => {
                    let struct_name = make::ext::ident_path("Self");
                    let fields = make::record_expr_field_list(None);
                    make::record_expr(struct_name, fields).into()
                }
            };
            let body = make::block_expr(None, Some(expr)).indent(ast::edit::IndentLevel(1));
            ted::replace(func.body()?.syntax(), body.clone_for_update().syntax());
            Some(())
        }
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
    fn add_custom_impl_debug_record_struct() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: fmt
#[derive(Debu$0g)]
struct Foo {
    bar: String,
}
"#,
            r#"
struct Foo {
    bar: String,
}

impl core::fmt::Debug for Foo {
    $0fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Foo").field("bar", &self.bar).finish()
    }
}
"#,
        )
    }
    #[test]
    fn add_custom_impl_debug_tuple_struct() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: fmt
#[derive(Debu$0g)]
struct Foo(String, usize);
"#,
            r#"struct Foo(String, usize);

impl core::fmt::Debug for Foo {
    $0fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Foo").field(&self.0).field(&self.1).finish()
    }
}
"#,
        )
    }
    #[test]
    fn add_custom_impl_debug_empty_struct() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: fmt
#[derive(Debu$0g)]
struct Foo;
"#,
            r#"
struct Foo;

impl core::fmt::Debug for Foo {
    $0fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Foo").finish()
    }
}
"#,
        )
    }
    #[test]
    fn add_custom_impl_debug_enum() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: fmt
#[derive(Debu$0g)]
enum Foo {
    Bar,
    Baz,
}
"#,
            r#"
enum Foo {
    Bar,
    Baz,
}

impl core::fmt::Debug for Foo {
    $0fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Bar => write!(f, "Bar"),
            Self::Baz => write!(f, "Baz"),
        }
    }
}
"#,
        )
    }
    #[test]
    fn add_custom_impl_default_record_struct() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: default
#[derive(Defau$0lt)]
struct Foo {
    foo: usize,
}
"#,
            r#"
struct Foo {
    foo: usize,
}

impl Default for Foo {
    $0fn default() -> Self {
        Self { foo: Default::default() }
    }
}
"#,
        )
    }
    #[test]
    fn add_custom_impl_default_tuple_struct() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: default
#[derive(Defau$0lt)]
struct Foo(usize);
"#,
            r#"
struct Foo(usize);

impl Default for Foo {
    $0fn default() -> Self {
        Self(Default::default())
    }
}
"#,
        )
    }
    #[test]
    fn add_custom_impl_default_empty_struct() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: default
#[derive(Defau$0lt)]
struct Foo;
"#,
            r#"
struct Foo;

impl Default for Foo {
    $0fn default() -> Self {
        Self {  }
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
