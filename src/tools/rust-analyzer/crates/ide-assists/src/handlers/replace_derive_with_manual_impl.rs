use hir::{InFile, MacroFileIdExt, ModuleDef};
use ide_db::{helpers::mod_path_to_ast, imports::import_assets::NameToImport, items_locator};
use itertools::Itertools;
use syntax::{
    ast::{self, make, AstNode, HasName},
    ted,
    SyntaxKind::WHITESPACE,
    T,
};

use crate::{
    assist_context::{AssistContext, Assists, SourceChangeBuilder},
    utils::{
        add_trait_assoc_items_to_impl, filter_assoc_items, gen_trait_fn_body, generate_trait_impl,
        DefaultMethods, IgnoreAssocItems,
    },
    AssistId, AssistKind,
};

// Assist: replace_derive_with_manual_impl
//
// Converts a `derive` impl into a manual one.
//
// ```
// # //- minicore: derive
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
    ctx: &AssistContext<'_>,
) -> Option<()> {
    let attr = ctx.find_node_at_offset_with_descend::<ast::Attr>()?;
    let path = attr.path()?;
    let macro_file = ctx.sema.hir_file_for(attr.syntax()).macro_file()?;
    if !macro_file.is_derive_attr_pseudo_expansion(ctx.db()) {
        return None;
    }

    let InFile { file_id, value } = macro_file.call_node(ctx.db());
    if file_id.is_macro() {
        // FIXME: make this work in macro files
        return None;
    }
    // collect the derive paths from the #[derive] expansion
    let current_derives = ctx
        .sema
        .parse_or_expand(macro_file.into())
        .descendants()
        .filter_map(ast::Attr::cast)
        .filter_map(|attr| attr.path())
        .collect::<Vec<_>>();

    let adt = value.parent().and_then(ast::Adt::cast)?;
    let attr = ast::Attr::cast(value)?;
    let args = attr.token_tree()?;

    let current_module = ctx.sema.scope(adt.syntax())?.module();
    let current_crate = current_module.krate();
    let current_edition = current_crate.edition(ctx.db());

    let found_traits = items_locator::items_with_name(
        &ctx.sema,
        current_crate,
        NameToImport::exact_case_sensitive(path.segments().last()?.to_string()),
        items_locator::AssocSearchMode::Exclude,
    )
    .filter_map(|item| match item.as_module_def()? {
        ModuleDef::Trait(trait_) => Some(trait_),
        _ => None,
    })
    .flat_map(|trait_| {
        current_module
            .find_path(ctx.sema.db, hir::ModuleDef::Trait(trait_), ctx.config.import_path_config())
            .as_ref()
            .map(|path| mod_path_to_ast(path, current_edition))
            .zip(Some(trait_))
    });

    let mut no_traits_found = true;
    for (replace_trait_path, trait_) in found_traits.inspect(|_| no_traits_found = false) {
        add_assist(
            acc,
            ctx,
            &attr,
            &current_derives,
            &args,
            &path,
            &replace_trait_path,
            Some(trait_),
            &adt,
        )?;
    }
    if no_traits_found {
        add_assist(acc, ctx, &attr, &current_derives, &args, &path, &path, None, &adt)?;
    }
    Some(())
}

fn add_assist(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
    attr: &ast::Attr,
    old_derives: &[ast::Path],
    old_tree: &ast::TokenTree,
    old_trait_path: &ast::Path,
    replace_trait_path: &ast::Path,
    trait_: Option<hir::Trait>,
    adt: &ast::Adt,
) -> Option<()> {
    let target = attr.syntax().text_range();
    let annotated_name = adt.name()?;
    let label = format!("Convert to manual `impl {replace_trait_path} for {annotated_name}`");

    acc.add(
        AssistId("replace_derive_with_manual_impl", AssistKind::Refactor),
        label,
        target,
        |builder| {
            let insert_after = ted::Position::after(builder.make_mut(adt.clone()).syntax());

            let impl_def_with_items =
                impl_def_from_trait(&ctx.sema, adt, &annotated_name, trait_, replace_trait_path);
            update_attribute(builder, old_derives, old_tree, old_trait_path, attr);

            let trait_path = make::ty_path(replace_trait_path.clone());

            match (ctx.config.snippet_cap, impl_def_with_items) {
                (None, _) => {
                    let impl_def = generate_trait_impl(adt, trait_path);

                    ted::insert_all(
                        insert_after,
                        vec![make::tokens::blank_line().into(), impl_def.syntax().clone().into()],
                    );
                }
                (Some(cap), None) => {
                    let impl_def = generate_trait_impl(adt, trait_path);

                    if let Some(l_curly) =
                        impl_def.assoc_item_list().and_then(|it| it.l_curly_token())
                    {
                        builder.add_tabstop_after_token(cap, l_curly);
                    }

                    ted::insert_all(
                        insert_after,
                        vec![make::tokens::blank_line().into(), impl_def.syntax().clone().into()],
                    );
                }
                (Some(cap), Some((impl_def, first_assoc_item))) => {
                    let mut added_snippet = false;
                    if let ast::AssocItem::Fn(ref func) = first_assoc_item {
                        if let Some(m) = func.syntax().descendants().find_map(ast::MacroCall::cast)
                        {
                            if m.syntax().text() == "todo!()" {
                                // Make the `todo!()` a placeholder
                                builder.add_placeholder_snippet(cap, m);
                                added_snippet = true;
                            }
                        }
                    }

                    if !added_snippet {
                        // If we haven't already added a snippet, add a tabstop before the generated function
                        builder.add_tabstop_before(cap, first_assoc_item);
                    }

                    ted::insert_all(
                        insert_after,
                        vec![make::tokens::blank_line().into(), impl_def.syntax().clone().into()],
                    );
                }
            };
        },
    )
}

fn impl_def_from_trait(
    sema: &hir::Semantics<'_, ide_db::RootDatabase>,
    adt: &ast::Adt,
    annotated_name: &ast::Name,
    trait_: Option<hir::Trait>,
    trait_path: &ast::Path,
) -> Option<(ast::Impl, ast::AssocItem)> {
    let trait_ = trait_?;
    let target_scope = sema.scope(annotated_name.syntax())?;

    // Keep assoc items of local crates even if they have #[doc(hidden)] attr.
    let ignore_items = if trait_.module(sema.db).krate().origin(sema.db).is_local() {
        IgnoreAssocItems::No
    } else {
        IgnoreAssocItems::DocHiddenAttrPresent
    };

    let trait_items =
        filter_assoc_items(sema, &trait_.items(sema.db), DefaultMethods::No, ignore_items);

    if trait_items.is_empty() {
        return None;
    }
    let impl_def = generate_trait_impl(adt, make::ty_path(trait_path.clone()));

    let first_assoc_item =
        add_trait_assoc_items_to_impl(sema, &trait_items, trait_, &impl_def, &target_scope);

    // Generate a default `impl` function body for the derived trait.
    if let ast::AssocItem::Fn(ref func) = first_assoc_item {
        let _ = gen_trait_fn_body(func, trait_path, adt, None);
    };

    Some((impl_def, first_assoc_item))
}

fn update_attribute(
    builder: &mut SourceChangeBuilder,
    old_derives: &[ast::Path],
    old_tree: &ast::TokenTree,
    old_trait_path: &ast::Path,
    attr: &ast::Attr,
) {
    let new_derives = old_derives
        .iter()
        .filter(|t| t.to_string() != old_trait_path.to_string())
        .collect::<Vec<_>>();
    let has_more_derives = !new_derives.is_empty();

    if has_more_derives {
        let old_tree = builder.make_mut(old_tree.clone());

        // Make the paths into flat lists of tokens in a vec
        let tt = new_derives.iter().map(|path| path.syntax().clone()).map(|node| {
            node.descendants_with_tokens()
                .filter_map(|element| element.into_token())
                .collect::<Vec<_>>()
        });
        // ...which are interspersed with ", "
        let tt = Itertools::intersperse(tt, vec![make::token(T![,]), make::tokens::single_space()]);
        // ...wrap them into the appropriate `NodeOrToken` variant
        let tt = tt.flatten().map(syntax::NodeOrToken::Token);
        // ...and make them into a flat list of tokens
        let tt = tt.collect::<Vec<_>>();

        let new_tree = make::token_tree(T!['('], tt).clone_for_update();
        ted::replace(old_tree.syntax(), new_tree.syntax());
    } else {
        // Remove the attr and any trailing whitespace
        let attr = builder.make_mut(attr.clone());

        if let Some(line_break) =
            attr.syntax().next_sibling_or_token().filter(|t| t.kind() == WHITESPACE)
        {
            ted::remove(line_break)
        }

        ted::remove(attr.syntax())
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
//- minicore: fmt, derive
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
//- minicore: fmt, derive
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
//- minicore: fmt, derive
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
//- minicore: fmt, derive
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
    fn add_custom_impl_debug_tuple_enum() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: fmt, derive
#[derive(Debu$0g)]
enum Foo {
    Bar(usize, usize),
    Baz,
}
"#,
            r#"
enum Foo {
    Bar(usize, usize),
    Baz,
}

impl core::fmt::Debug for Foo {
    $0fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Bar(arg0, arg1) => f.debug_tuple("Bar").field(arg0).field(arg1).finish(),
            Self::Baz => write!(f, "Baz"),
        }
    }
}
"#,
        )
    }
    #[test]
    fn add_custom_impl_debug_record_enum() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: fmt, derive
#[derive(Debu$0g)]
enum Foo {
    Bar {
        baz: usize,
        qux: usize,
    },
    Baz,
}
"#,
            r#"
enum Foo {
    Bar {
        baz: usize,
        qux: usize,
    },
    Baz,
}

impl core::fmt::Debug for Foo {
    $0fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Bar { baz, qux } => f.debug_struct("Bar").field("baz", baz).field("qux", qux).finish(),
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
//- minicore: default, derive
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
//- minicore: default, derive
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
//- minicore: default, derive
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
    fn add_custom_impl_hash_record_struct() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: hash, derive
#[derive(Has$0h)]
struct Foo {
    bin: usize,
    bar: usize,
}
"#,
            r#"
struct Foo {
    bin: usize,
    bar: usize,
}

impl core::hash::Hash for Foo {
    $0fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.bin.hash(state);
        self.bar.hash(state);
    }
}
"#,
        )
    }

    #[test]
    fn add_custom_impl_hash_tuple_struct() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: hash, derive
#[derive(Has$0h)]
struct Foo(usize, usize);
"#,
            r#"
struct Foo(usize, usize);

impl core::hash::Hash for Foo {
    $0fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
        self.1.hash(state);
    }
}
"#,
        )
    }

    #[test]
    fn add_custom_impl_hash_enum() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: hash, derive
#[derive(Has$0h)]
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

impl core::hash::Hash for Foo {
    $0fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
    }
}
"#,
        )
    }

    #[test]
    fn add_custom_impl_clone_record_struct() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: clone, derive
#[derive(Clo$0ne)]
struct Foo {
    bin: usize,
    bar: usize,
}
"#,
            r#"
struct Foo {
    bin: usize,
    bar: usize,
}

impl Clone for Foo {
    $0fn clone(&self) -> Self {
        Self { bin: self.bin.clone(), bar: self.bar.clone() }
    }
}
"#,
        )
    }

    #[test]
    fn add_custom_impl_clone_tuple_struct() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: clone, derive
#[derive(Clo$0ne)]
struct Foo(usize, usize);
"#,
            r#"
struct Foo(usize, usize);

impl Clone for Foo {
    $0fn clone(&self) -> Self {
        Self(self.0.clone(), self.1.clone())
    }
}
"#,
        )
    }

    #[test]
    fn add_custom_impl_clone_empty_struct() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: clone, derive
#[derive(Clo$0ne)]
struct Foo;
"#,
            r#"
struct Foo;

impl Clone for Foo {
    $0fn clone(&self) -> Self {
        Self {  }
    }
}
"#,
        )
    }

    #[test]
    fn add_custom_impl_clone_enum() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: clone, derive
#[derive(Clo$0ne)]
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

impl Clone for Foo {
    $0fn clone(&self) -> Self {
        match self {
            Self::Bar => Self::Bar,
            Self::Baz => Self::Baz,
        }
    }
}
"#,
        )
    }

    #[test]
    fn add_custom_impl_clone_tuple_enum() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: clone, derive
#[derive(Clo$0ne)]
enum Foo {
    Bar(String),
    Baz,
}
"#,
            r#"
enum Foo {
    Bar(String),
    Baz,
}

impl Clone for Foo {
    $0fn clone(&self) -> Self {
        match self {
            Self::Bar(arg0) => Self::Bar(arg0.clone()),
            Self::Baz => Self::Baz,
        }
    }
}
"#,
        )
    }

    #[test]
    fn add_custom_impl_clone_record_enum() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: clone, derive
#[derive(Clo$0ne)]
enum Foo {
    Bar {
        bin: String,
    },
    Baz,
}
"#,
            r#"
enum Foo {
    Bar {
        bin: String,
    },
    Baz,
}

impl Clone for Foo {
    $0fn clone(&self) -> Self {
        match self {
            Self::Bar { bin } => Self::Bar { bin: bin.clone() },
            Self::Baz => Self::Baz,
        }
    }
}
"#,
        )
    }

    #[test]
    fn add_custom_impl_partial_ord_record_struct() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: ord, derive
#[derive(Partial$0Ord)]
struct Foo {
    bin: usize,
}
"#,
            r#"
struct Foo {
    bin: usize,
}

impl PartialOrd for Foo {
    $0fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.bin.partial_cmp(&other.bin)
    }
}
"#,
        )
    }

    #[test]
    fn add_custom_impl_partial_ord_record_struct_multi_field() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: ord, derive
#[derive(Partial$0Ord)]
struct Foo {
    bin: usize,
    bar: usize,
    baz: usize,
}
"#,
            r#"
struct Foo {
    bin: usize,
    bar: usize,
    baz: usize,
}

impl PartialOrd for Foo {
    $0fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        match self.bin.partial_cmp(&other.bin) {
            Some(core::cmp::Ordering::Equal) => {}
            ord => return ord,
        }
        match self.bar.partial_cmp(&other.bar) {
            Some(core::cmp::Ordering::Equal) => {}
            ord => return ord,
        }
        self.baz.partial_cmp(&other.baz)
    }
}
"#,
        )
    }

    #[test]
    fn add_custom_impl_partial_ord_tuple_struct() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: ord, derive
#[derive(Partial$0Ord)]
struct Foo(usize, usize, usize);
"#,
            r#"
struct Foo(usize, usize, usize);

impl PartialOrd for Foo {
    $0fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        match self.0.partial_cmp(&other.0) {
            Some(core::cmp::Ordering::Equal) => {}
            ord => return ord,
        }
        match self.1.partial_cmp(&other.1) {
            Some(core::cmp::Ordering::Equal) => {}
            ord => return ord,
        }
        self.2.partial_cmp(&other.2)
    }
}
"#,
        )
    }

    #[test]
    fn add_custom_impl_partial_eq_record_struct() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: eq, derive
#[derive(Partial$0Eq)]
struct Foo {
    bin: usize,
    bar: usize,
}
"#,
            r#"
struct Foo {
    bin: usize,
    bar: usize,
}

impl PartialEq for Foo {
    $0fn eq(&self, other: &Self) -> bool {
        self.bin == other.bin && self.bar == other.bar
    }
}
"#,
        )
    }

    #[test]
    fn add_custom_impl_partial_eq_tuple_struct() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: eq, derive
#[derive(Partial$0Eq)]
struct Foo(usize, usize);
"#,
            r#"
struct Foo(usize, usize);

impl PartialEq for Foo {
    $0fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}
"#,
        )
    }

    #[test]
    fn add_custom_impl_partial_eq_empty_struct() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: eq, derive
#[derive(Partial$0Eq)]
struct Foo;
"#,
            r#"
struct Foo;

impl PartialEq for Foo {
    $0fn eq(&self, other: &Self) -> bool {
        true
    }
}
"#,
        )
    }

    #[test]
    fn add_custom_impl_partial_eq_enum() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: eq, derive
#[derive(Partial$0Eq)]
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

impl PartialEq for Foo {
    $0fn eq(&self, other: &Self) -> bool {
        core::mem::discriminant(self) == core::mem::discriminant(other)
    }
}
"#,
        )
    }

    #[test]
    fn add_custom_impl_partial_eq_single_variant_tuple_enum() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: eq, derive
#[derive(Partial$0Eq)]
enum Foo {
    Bar(String),
}
"#,
            r#"
enum Foo {
    Bar(String),
}

impl PartialEq for Foo {
    $0fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Bar(l0), Self::Bar(r0)) => l0 == r0,
        }
    }
}
"#,
        )
    }

    #[test]
    fn add_custom_impl_partial_eq_partial_tuple_enum() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: eq, derive
#[derive(Partial$0Eq)]
enum Foo {
    Bar(String),
    Baz,
}
"#,
            r#"
enum Foo {
    Bar(String),
    Baz,
}

impl PartialEq for Foo {
    $0fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Bar(l0), Self::Bar(r0)) => l0 == r0,
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}
"#,
        )
    }

    #[test]
    fn add_custom_impl_partial_eq_tuple_enum() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: eq, derive
#[derive(Partial$0Eq)]
enum Foo {
    Bar(String),
    Baz(i32),
}
"#,
            r#"
enum Foo {
    Bar(String),
    Baz(i32),
}

impl PartialEq for Foo {
    $0fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Bar(l0), Self::Bar(r0)) => l0 == r0,
            (Self::Baz(l0), Self::Baz(r0)) => l0 == r0,
            _ => false,
        }
    }
}
"#,
        )
    }

    #[test]
    fn add_custom_impl_partial_eq_tuple_enum_generic() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: eq, derive
#[derive(Partial$0Eq)]
enum Either<T, U> {
    Left(T),
    Right(U),
}
"#,
            r#"
enum Either<T, U> {
    Left(T),
    Right(U),
}

impl<T: PartialEq, U: PartialEq> PartialEq for Either<T, U> {
    $0fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Left(l0), Self::Left(r0)) => l0 == r0,
            (Self::Right(l0), Self::Right(r0)) => l0 == r0,
            _ => false,
        }
    }
}
"#,
        )
    }

    #[test]
    fn add_custom_impl_partial_eq_tuple_enum_generic_existing_bounds() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: eq, derive
#[derive(Partial$0Eq)]
enum Either<T: PartialEq + Error, U: Clone> {
    Left(T),
    Right(U),
}
"#,
            r#"
enum Either<T: PartialEq + Error, U: Clone> {
    Left(T),
    Right(U),
}

impl<T: PartialEq + Error, U: Clone + PartialEq> PartialEq for Either<T, U> {
    $0fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Left(l0), Self::Left(r0)) => l0 == r0,
            (Self::Right(l0), Self::Right(r0)) => l0 == r0,
            _ => false,
        }
    }
}
"#,
        )
    }

    #[test]
    fn add_custom_impl_partial_eq_record_enum() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: eq, derive
#[derive(Partial$0Eq)]
enum Foo {
    Bar {
        bin: String,
    },
    Baz {
        qux: String,
        fez: String,
    },
    Qux {},
    Bin,
}
"#,
            r#"
enum Foo {
    Bar {
        bin: String,
    },
    Baz {
        qux: String,
        fez: String,
    },
    Qux {},
    Bin,
}

impl PartialEq for Foo {
    $0fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Bar { bin: l_bin }, Self::Bar { bin: r_bin }) => l_bin == r_bin,
            (Self::Baz { qux: l_qux, fez: l_fez }, Self::Baz { qux: r_qux, fez: r_fez }) => l_qux == r_qux && l_fez == r_fez,
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
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
//- minicore: derive
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

    const Fez: usize;

    fn foo() {
        todo!()
    }
}
"#,
        )
    }
    #[test]
    fn add_custom_impl_for_unique_input_unknown() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: derive
#[derive(Debu$0g)]
struct Foo {
    bar: String,
}
            "#,
            r#"
struct Foo {
    bar: String,
}

impl Debug for Foo {$0}
            "#,
        )
    }

    #[test]
    fn add_custom_impl_for_with_visibility_modifier() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: derive
#[derive(Debug$0)]
pub struct Foo {
    bar: String,
}
            "#,
            r#"
pub struct Foo {
    bar: String,
}

impl Debug for Foo {$0}
            "#,
        )
    }

    #[test]
    fn add_custom_impl_when_multiple_inputs() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: derive
#[derive(Display, Debug$0, Serialize)]
struct Foo {}
            "#,
            r#"
#[derive(Display, Serialize)]
struct Foo {}

impl Debug for Foo {$0}
            "#,
        )
    }

    #[test]
    fn add_custom_impl_default_generic_record_struct() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: default, derive
#[derive(Defau$0lt)]
struct Foo<T, U> {
    foo: T,
    bar: U,
}
"#,
            r#"
struct Foo<T, U> {
    foo: T,
    bar: U,
}

impl<T: Default, U: Default> Default for Foo<T, U> {
    $0fn default() -> Self {
        Self { foo: Default::default(), bar: Default::default() }
    }
}
"#,
        )
    }

    #[test]
    fn add_custom_impl_clone_generic_tuple_struct_with_bounds() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: clone, derive
#[derive(Clo$0ne)]
struct Foo<T: Clone>(T, usize);
"#,
            r#"
struct Foo<T: Clone>(T, usize);

impl<T: Clone> Clone for Foo<T> {
    $0fn clone(&self) -> Self {
        Self(self.0.clone(), self.1.clone())
    }
}
"#,
        )
    }

    #[test]
    fn test_ignore_derive_macro_without_input() {
        check_assist_not_applicable(
            replace_derive_with_manual_impl,
            r#"
//- minicore: derive
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
//- minicore: derive, fmt
#[derive$0(Debug)]
struct Foo {}
            "#,
        );

        check_assist_not_applicable(
            replace_derive_with_manual_impl,
            r#"
//- minicore: derive, fmt
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
//- minicore: derive
#[allow(non_camel_$0case_types)]
struct Foo {}
            "#,
        )
    }

    #[test]
    fn works_at_start_of_file() {
        check_assist_not_applicable(
            replace_derive_with_manual_impl,
            r#"
//- minicore: derive, fmt
$0#[derive(Debug)]
struct S;
            "#,
        );
    }

    #[test]
    fn add_custom_impl_keep_path() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: clone, derive
#[derive(std::fmt::Debug, Clo$0ne)]
pub struct Foo;
"#,
            r#"
#[derive(std::fmt::Debug)]
pub struct Foo;

impl Clone for Foo {
    $0fn clone(&self) -> Self {
        Self {  }
    }
}
"#,
        )
    }

    #[test]
    fn add_custom_impl_replace_path() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: fmt, derive
#[derive(core::fmt::Deb$0ug, Clone)]
pub struct Foo;
"#,
            r#"
#[derive(Clone)]
pub struct Foo;

impl core::fmt::Debug for Foo {
    $0fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Foo").finish()
    }
}
"#,
        )
    }
}
