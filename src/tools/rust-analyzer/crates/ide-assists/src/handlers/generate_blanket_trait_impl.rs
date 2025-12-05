use crate::{
    AssistConfig,
    assist_context::{AssistContext, Assists},
};
use hir::{HasCrate, Semantics};
use ide_db::{
    RootDatabase,
    assists::{AssistId, AssistKind, ExprFillDefaultMode},
    famous_defs::FamousDefs,
    syntax_helpers::suggest_name,
};
use syntax::{
    AstNode,
    ast::{
        self, AssocItem, BlockExpr, GenericParam, HasAttrs, HasGenericParams, HasName,
        HasTypeBounds, HasVisibility, edit::AstNodeEdit, make,
    },
    syntax_editor::Position,
};

// Assist: generate_blanket_trait_impl
//
// Generate blanket trait implementation.
//
// ```
// trait $0Foo<T: Send>: ToOwned
// where
//     Self::Owned: Default,
// {
//     fn foo(&self) -> T;
//
//     fn print_foo(&self) {
//         println!("{}", self.foo());
//     }
// }
// ```
// ->
// ```
// trait Foo<T: Send>: ToOwned
// where
//     Self::Owned: Default,
// {
//     fn foo(&self) -> T;
//
//     fn print_foo(&self) {
//         println!("{}", self.foo());
//     }
// }
//
// impl<T: Send, T1: ToOwned + ?Sized> Foo<T> for $0T1
// where
//     Self::Owned: Default,
// {
//     fn foo(&self) -> T {
//         todo!()
//     }
// }
// ```
pub(crate) fn generate_blanket_trait_impl(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
) -> Option<()> {
    let name = ctx.find_node_at_offset::<ast::Name>()?;
    let traitd = ast::Trait::cast(name.syntax().parent()?)?;

    if existing_any_impl(&traitd, &ctx.sema).is_some() {
        cov_mark::hit!(existing_any_impl);
        return None;
    }

    acc.add(
        AssistId("generate_blanket_trait_impl", AssistKind::Generate, None),
        "Generate blanket trait implementation",
        name.syntax().text_range(),
        |builder| {
            let mut edit = builder.make_editor(traitd.syntax());
            let namety = make::ty_path(make::path_from_text(&name.text()));
            let trait_where_clause = traitd.where_clause().map(|it| it.reset_indent());
            let bounds = traitd.type_bound_list().and_then(exlucde_sized);
            let is_unsafe = traitd.unsafe_token().is_some();
            let thisname = this_name(&traitd);
            let thisty = make::ty_path(make::path_from_text(&thisname.text()));
            let indent = traitd.indent_level();

            let gendecl = make::generic_param_list([GenericParam::TypeParam(make::type_param(
                thisname.clone(),
                apply_sized(has_sized(&traitd, &ctx.sema), bounds),
            ))]);

            let trait_gen_args =
                traitd.generic_param_list().map(|param_list| param_list.to_generic_args());

            let impl_ = make::impl_trait(
                cfg_attrs(&traitd),
                is_unsafe,
                traitd.generic_param_list(),
                trait_gen_args,
                Some(gendecl),
                None,
                false,
                namety,
                thisty.clone(),
                trait_where_clause,
                None,
                None,
            )
            .clone_for_update();

            if let Some(trait_assoc_list) = traitd.assoc_item_list() {
                let assoc_item_list = impl_.get_or_create_assoc_item_list();
                for item in trait_assoc_list.assoc_items() {
                    let item = match item {
                        ast::AssocItem::Fn(method) if method.body().is_none() => {
                            todo_fn(&method, ctx.config).into()
                        }
                        ast::AssocItem::Const(_) | ast::AssocItem::TypeAlias(_) => item,
                        _ => continue,
                    };
                    assoc_item_list.add_item(item.reset_indent().indent(1.into()));
                }
            }

            let impl_ = impl_.indent(indent);

            edit.insert_all(
                Position::after(traitd.syntax()),
                vec![
                    make::tokens::whitespace(&format!("\n\n{indent}")).into(),
                    impl_.syntax().clone().into(),
                ],
            );

            if let Some(cap) = ctx.config.snippet_cap
                && let Some(self_ty) = impl_.self_ty()
            {
                builder.add_tabstop_before(cap, self_ty);
            }

            builder.add_file_edits(ctx.vfs_file_id(), edit);
        },
    );

    Some(())
}

fn existing_any_impl(traitd: &ast::Trait, sema: &Semantics<'_, RootDatabase>) -> Option<hir::Impl> {
    let db = sema.db;
    let traitd = sema.to_def(traitd)?;
    traitd
        .module(db)
        .impl_defs(db)
        .into_iter()
        .find(|impl_| impl_.trait_(db).is_some_and(|it| it == traitd))
}

fn has_sized(traitd: &ast::Trait, sema: &Semantics<'_, RootDatabase>) -> bool {
    if let Some(sized) = find_bound("Sized", traitd.type_bound_list()) {
        sized.question_mark_token().is_none()
    } else if let Some(is_sized) = where_clause_sized(traitd.where_clause()) {
        is_sized
    } else {
        contained_owned_self_method(traitd.assoc_item_list())
            || super_traits_has_sized(traitd, sema) == Some(true)
    }
}

fn super_traits_has_sized(traitd: &ast::Trait, sema: &Semantics<'_, RootDatabase>) -> Option<bool> {
    let traitd = sema.to_def(traitd)?;
    let sized = FamousDefs(sema, traitd.krate(sema.db)).core_marker_Sized()?;

    Some(traitd.all_supertraits(sema.db).contains(&sized))
}

fn contained_owned_self_method(item_list: Option<ast::AssocItemList>) -> bool {
    item_list.into_iter().flat_map(|assoc_item_list| assoc_item_list.assoc_items()).any(|item| {
        match item {
            AssocItem::Fn(f) => {
                has_owned_self(&f) && where_clause_sized(f.where_clause()).is_none()
            }
            _ => false,
        }
    })
}

fn has_owned_self(f: &ast::Fn) -> bool {
    has_owned_self_param(f) || has_ret_owned_self(f)
}

fn has_owned_self_param(f: &ast::Fn) -> bool {
    f.param_list()
        .and_then(|param_list| param_list.self_param())
        .is_some_and(|sp| sp.amp_token().is_none() && sp.colon_token().is_none())
}

fn has_ret_owned_self(f: &ast::Fn) -> bool {
    f.ret_type()
        .and_then(|ret| match ret.ty() {
            Some(ast::Type::PathType(ty)) => ty.path(),
            _ => None,
        })
        .is_some_and(|path| {
            path.segment()
                .and_then(|seg| seg.name_ref())
                .is_some_and(|name| path.qualifier().is_none() && name.text() == "Self")
        })
}

fn where_clause_sized(where_clause: Option<ast::WhereClause>) -> Option<bool> {
    where_clause?.predicates().find_map(|pred| {
        find_bound("Sized", pred.type_bound_list())
            .map(|bound| bound.question_mark_token().is_none())
    })
}

fn apply_sized(has_sized: bool, bounds: Option<ast::TypeBoundList>) -> Option<ast::TypeBoundList> {
    if has_sized {
        return bounds;
    }
    let bounds = bounds
        .into_iter()
        .flat_map(|bounds| bounds.bounds())
        .chain([make::type_bound_text("?Sized")]);
    make::type_bound_list(bounds)
}

fn exlucde_sized(bounds: ast::TypeBoundList) -> Option<ast::TypeBoundList> {
    make::type_bound_list(bounds.bounds().filter(|bound| !ty_bound_is(bound, "Sized")))
}

fn this_name(traitd: &ast::Trait) -> ast::Name {
    let has_iter = find_bound("Iterator", traitd.type_bound_list()).is_some();

    let params = traitd
        .generic_param_list()
        .into_iter()
        .flat_map(|param_list| param_list.generic_params())
        .filter_map(|param| match param {
            GenericParam::LifetimeParam(_) => None,
            GenericParam::ConstParam(cp) => cp.name(),
            GenericParam::TypeParam(tp) => tp.name(),
        })
        .map(|name| name.to_string())
        .collect::<Vec<_>>();

    let mut name_gen =
        suggest_name::NameGenerator::new_with_names(params.iter().map(String::as_str));

    make::name(&name_gen.suggest_name(if has_iter { "I" } else { "T" }))
}

fn find_bound(s: &str, bounds: Option<ast::TypeBoundList>) -> Option<ast::TypeBound> {
    bounds.into_iter().flat_map(|bounds| bounds.bounds()).find(|bound| ty_bound_is(bound, s))
}

fn ty_bound_is(bound: &ast::TypeBound, s: &str) -> bool {
    matches!(bound.ty(),
        Some(ast::Type::PathType(ty)) if ty.path()
            .and_then(|path| path.segment())
            .and_then(|segment| segment.name_ref())
            .is_some_and(|name| name.text() == s))
}

fn todo_fn(f: &ast::Fn, config: &AssistConfig) -> ast::Fn {
    let params = f.param_list().unwrap_or_else(|| make::param_list(None, None));
    make::fn_(
        cfg_attrs(f),
        f.visibility(),
        f.name().unwrap_or_else(|| make::name("unnamed")),
        f.generic_param_list(),
        f.where_clause(),
        params,
        default_block(config),
        f.ret_type(),
        f.async_token().is_some(),
        f.const_token().is_some(),
        f.unsafe_token().is_some(),
        f.gen_token().is_some(),
    )
}

fn default_block(config: &AssistConfig) -> BlockExpr {
    let expr = match config.expr_fill_default {
        ExprFillDefaultMode::Todo => make::ext::expr_todo(),
        ExprFillDefaultMode::Underscore => make::ext::expr_underscore(),
        ExprFillDefaultMode::Default => make::ext::expr_todo(),
    };
    make::block_expr(None, Some(expr))
}

fn cfg_attrs(node: &impl HasAttrs) -> impl Iterator<Item = ast::Attr> {
    node.attrs().filter(|attr| attr.as_simple_call().is_some_and(|(name, _arg)| name == "cfg"))
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::tests::{check_assist, check_assist_not_applicable};

    #[test]
    fn test_gen_blanket_works() {
        check_assist(
            generate_blanket_trait_impl,
            r#"
trait $0Foo<T: Send>: ToOwned
where
    Self::Owned: Default,
{
    fn foo(&self) -> T;

    fn print_foo(&self) {
        println!("{}", self.foo());
    }
}
"#,
            r#"
trait Foo<T: Send>: ToOwned
where
    Self::Owned: Default,
{
    fn foo(&self) -> T;

    fn print_foo(&self) {
        println!("{}", self.foo());
    }
}

impl<T: Send, T1: ToOwned + ?Sized> Foo<T> for $0T1
where
    Self::Owned: Default,
{
    fn foo(&self) -> T {
        todo!()
    }
}
"#,
        );
    }

    #[test]
    fn test_gen_blanket_sized() {
        check_assist(
            generate_blanket_trait_impl,
            r#"
trait $0Foo: Iterator + Sized {
    fn foo(mut self) -> Self::Item {
        self.next().unwrap()
    }
}
"#,
            r#"
trait Foo: Iterator + Sized {
    fn foo(mut self) -> Self::Item {
        self.next().unwrap()
    }
}

impl<I: Iterator> Foo for $0I {}
"#,
        );

        check_assist(
            generate_blanket_trait_impl,
            r#"
trait $0Foo: Iterator {
    fn foo(self) -> Self::Item;
}
"#,
            r#"
trait Foo: Iterator {
    fn foo(self) -> Self::Item;
}

impl<I: Iterator> Foo for $0I {
    fn foo(self) -> Self::Item {
        todo!()
    }
}
"#,
        );

        check_assist(
            generate_blanket_trait_impl,
            r#"
trait $0Foo {
    fn foo(&self) -> Self;
}
"#,
            r#"
trait Foo {
    fn foo(&self) -> Self;
}

impl<T> Foo for $0T {
    fn foo(&self) -> Self {
        todo!()
    }
}
"#,
        );
    }

    #[test]
    fn test_gen_blanket_super_sized() {
        check_assist(
            generate_blanket_trait_impl,
            r#"
//- minicore: default
trait $0Foo: Default {
    fn foo(&self);
}
"#,
            r#"
trait Foo: Default {
    fn foo(&self);
}

impl<T: Default> Foo for $0T {
    fn foo(&self) {
        todo!()
    }
}
"#,
        );
    }

    #[test]
    fn test_gen_blanket_non_sized() {
        check_assist(
            generate_blanket_trait_impl,
            r#"
trait $0Foo: Iterator {
    fn foo(&self) -> Self::Item;
}
"#,
            r#"
trait Foo: Iterator {
    fn foo(&self) -> Self::Item;
}

impl<I: Iterator + ?Sized> Foo for $0I {
    fn foo(&self) -> Self::Item {
        todo!()
    }
}
"#,
        );
        check_assist(
            generate_blanket_trait_impl,
            r#"
trait $0Foo: Iterator {
    fn foo(&self) -> Self::Item;

    fn each(self) where Self: Sized;
}
"#,
            r#"
trait Foo: Iterator {
    fn foo(&self) -> Self::Item;

    fn each(self) where Self: Sized;
}

impl<I: Iterator + ?Sized> Foo for $0I {
    fn foo(&self) -> Self::Item {
        todo!()
    }

    fn each(self) where Self: Sized {
        todo!()
    }
}
"#,
        );
        check_assist(
            generate_blanket_trait_impl,
            r#"
trait $0Foo: Iterator {
    fn foo(&self) -> Self::Item;

    fn each(&self) -> Self where Self: Sized;
}
"#,
            r#"
trait Foo: Iterator {
    fn foo(&self) -> Self::Item;

    fn each(&self) -> Self where Self: Sized;
}

impl<I: Iterator + ?Sized> Foo for $0I {
    fn foo(&self) -> Self::Item {
        todo!()
    }

    fn each(&self) -> Self where Self: Sized {
        todo!()
    }
}
"#,
        );
    }

    #[test]
    fn test_gen_blanket_other_assoc_items() {
        check_assist(
            generate_blanket_trait_impl,
            r#"
trait $0Foo {
    type Item;

    const N: usize;

    fn foo(&self);
}
"#,
            r#"
trait Foo {
    type Item;

    const N: usize;

    fn foo(&self);
}

impl<T: ?Sized> Foo for $0T {
    type Item;

    const N: usize;

    fn foo(&self) {
        todo!()
    }
}
"#,
        );
    }

    #[test]
    fn test_gen_blanket_indent() {
        check_assist(
            generate_blanket_trait_impl,
            r#"
mod foo {
    trait $0Foo<T: Send>: ToOwned
    where
        Self::Owned: Default,
    {
        fn foo(&self) -> T;

        fn print_foo(&self) {
            println!("{}", self.foo());
        }
    }
}
        "#,
            r#"
mod foo {
    trait Foo<T: Send>: ToOwned
    where
        Self::Owned: Default,
    {
        fn foo(&self) -> T;

        fn print_foo(&self) {
            println!("{}", self.foo());
        }
    }

    impl<T: Send, T1: ToOwned + ?Sized> Foo<T> for $0T1
    where
        Self::Owned: Default,
    {
        fn foo(&self) -> T {
            todo!()
        }
    }
}
        "#,
        );
        check_assist(
            generate_blanket_trait_impl,
            r#"
mod foo {
    trait $0Foo<T: Send>: ToOwned
    where
        Self::Owned: Default,
        Self: Send,
    {
        fn foo(&self) -> T;

        fn print_foo(&self) {
            println!("{}", self.foo());
        }
    }
}
        "#,
            r#"
mod foo {
    trait Foo<T: Send>: ToOwned
    where
        Self::Owned: Default,
        Self: Send,
    {
        fn foo(&self) -> T;

        fn print_foo(&self) {
            println!("{}", self.foo());
        }
    }

    impl<T: Send, T1: ToOwned + ?Sized> Foo<T> for $0T1
    where
        Self::Owned: Default,
        Self: Send,
    {
        fn foo(&self) -> T {
            todo!()
        }
    }
}
        "#,
        );
        check_assist(
            generate_blanket_trait_impl,
            r#"
mod foo {
    mod bar {
        trait $0Foo<T: Send>: ToOwned
        where
            Self::Owned: Default,
            Self: Send,
        {
            fn foo(&self) -> T;

            fn print_foo(&self) {
                println!("{}", self.foo());
            }
        }
    }
}
        "#,
            r#"
mod foo {
    mod bar {
        trait Foo<T: Send>: ToOwned
        where
            Self::Owned: Default,
            Self: Send,
        {
            fn foo(&self) -> T;

            fn print_foo(&self) {
                println!("{}", self.foo());
            }
        }

        impl<T: Send, T1: ToOwned + ?Sized> Foo<T> for $0T1
        where
            Self::Owned: Default,
            Self: Send,
        {
            fn foo(&self) -> T {
                todo!()
            }
        }
    }
}
        "#,
        );
        check_assist(
            generate_blanket_trait_impl,
            r#"
mod foo {
    trait $0Foo {
        fn foo(&self) -> i32;

        fn print_foo(&self) {
            println!("{}", self.foo());
        }
    }
}
        "#,
            r#"
mod foo {
    trait Foo {
        fn foo(&self) -> i32;

        fn print_foo(&self) {
            println!("{}", self.foo());
        }
    }

    impl<T: ?Sized> Foo for $0T {
        fn foo(&self) -> i32 {
            todo!()
        }
    }
}
        "#,
        );
        check_assist(
            generate_blanket_trait_impl,
            r#"
mod foo {
    mod bar {
        trait $0Foo {
            fn foo(&self) -> i32;

            fn print_foo(&self) {
                println!("{}", self.foo());
            }
        }
    }
}
        "#,
            r#"
mod foo {
    mod bar {
        trait Foo {
            fn foo(&self) -> i32;

            fn print_foo(&self) {
                println!("{}", self.foo());
            }
        }

        impl<T: ?Sized> Foo for $0T {
            fn foo(&self) -> i32 {
                todo!()
            }
        }
    }
}
        "#,
        );
        check_assist(
            generate_blanket_trait_impl,
            r#"
mod foo {
    mod bar {
        #[cfg(test)]
        trait $0Foo {
            fn foo(&self) -> i32;

            fn print_foo(&self) {
                println!("{}", self.foo());
            }
        }
    }
}
        "#,
            r#"
mod foo {
    mod bar {
        #[cfg(test)]
        trait Foo {
            fn foo(&self) -> i32;

            fn print_foo(&self) {
                println!("{}", self.foo());
            }
        }

        #[cfg(test)]
        impl<T: ?Sized> Foo for $0T {
            fn foo(&self) -> i32 {
                todo!()
            }
        }
    }
}
        "#,
        );
        check_assist(
            generate_blanket_trait_impl,
            r#"
mod foo {
    mod bar {
        trait $0Foo {
            type Item: Bar<
                Self,
            >;

            const N: Baz<
                Self,
            >;
        }
    }
}
        "#,
            r#"
mod foo {
    mod bar {
        trait Foo {
            type Item: Bar<
                Self,
            >;

            const N: Baz<
                Self,
            >;
        }

        impl<T: ?Sized> Foo for $0T {
            type Item: Bar<
                Self,
            >;

            const N: Baz<
                Self,
            >;
        }
    }
}
        "#,
        );
    }

    #[test]
    fn test_gen_blanket_remove_attribute() {
        check_assist(
            generate_blanket_trait_impl,
            r#"
trait $0Foo<T: Send>: ToOwned
where
    Self::Owned: Default,
{
    #[doc(hidden)]
    fn foo(&self) -> T;

    /// foo
    fn print_foo(&self) {
        println!("{}", self.foo());
    }
}
"#,
            r#"
trait Foo<T: Send>: ToOwned
where
    Self::Owned: Default,
{
    #[doc(hidden)]
    fn foo(&self) -> T;

    /// foo
    fn print_foo(&self) {
        println!("{}", self.foo());
    }
}

impl<T: Send, T1: ToOwned + ?Sized> Foo<T> for $0T1
where
    Self::Owned: Default,
{
    fn foo(&self) -> T {
        todo!()
    }
}
"#,
        );
    }

    #[test]
    fn test_gen_blanket_not_gen_type_alias() {
        check_assist(
            generate_blanket_trait_impl,
            r#"
trait $0Foo<T: Send>: ToOwned
where
    Self::Owned: Default,
{
    type X: Sync;

    fn foo(&self, x: Self::X) -> T;

    fn print_foo(&self) {
        println!("{}", self.foo());
    }
}
"#,
            r#"
trait Foo<T: Send>: ToOwned
where
    Self::Owned: Default,
{
    type X: Sync;

    fn foo(&self, x: Self::X) -> T;

    fn print_foo(&self) {
        println!("{}", self.foo());
    }
}

impl<T: Send, T1: ToOwned + ?Sized> Foo<T> for $0T1
where
    Self::Owned: Default,
{
    type X: Sync;

    fn foo(&self, x: Self::X) -> T {
        todo!()
    }
}
"#,
        );
    }

    #[test]
    fn test_gen_blanket_no_quick_bound() {
        check_assist(
            generate_blanket_trait_impl,
            r#"
trait $0Foo<T: Send>
where
    Self: ToOwned,
    Self::Owned: Default,
{
    type X: Sync;

    fn foo(&self, x: Self::X) -> T;

    fn print_foo(&self) {
        println!("{}", self.foo());
    }
}
"#,
            r#"
trait Foo<T: Send>
where
    Self: ToOwned,
    Self::Owned: Default,
{
    type X: Sync;

    fn foo(&self, x: Self::X) -> T;

    fn print_foo(&self) {
        println!("{}", self.foo());
    }
}

impl<T: Send, T1: ?Sized> Foo<T> for $0T1
where
    Self: ToOwned,
    Self::Owned: Default,
{
    type X: Sync;

    fn foo(&self, x: Self::X) -> T {
        todo!()
    }
}
"#,
        );
    }

    #[test]
    fn test_gen_blanket_no_where_clause() {
        check_assist(
            generate_blanket_trait_impl,
            r#"
trait $0Foo<T: Send> {
    type X: Sync;

    fn foo(&self, x: Self::X) -> T;

    fn print_foo(&self) {
        println!("{}", self.foo());
    }
}
"#,
            r#"
trait Foo<T: Send> {
    type X: Sync;

    fn foo(&self, x: Self::X) -> T;

    fn print_foo(&self) {
        println!("{}", self.foo());
    }
}

impl<T: Send, T1: ?Sized> Foo<T> for $0T1 {
    type X: Sync;

    fn foo(&self, x: Self::X) -> T {
        todo!()
    }
}
"#,
        );
    }

    #[test]
    fn test_gen_blanket_basic() {
        check_assist(
            generate_blanket_trait_impl,
            r#"
trait $0Foo {
    type X: Sync;

    fn foo(&self, x: Self::X) -> i32;

    fn print_foo(&self) {
        println!("{}", self.foo());
    }
}
"#,
            r#"
trait Foo {
    type X: Sync;

    fn foo(&self, x: Self::X) -> i32;

    fn print_foo(&self) {
        println!("{}", self.foo());
    }
}

impl<T: ?Sized> Foo for $0T {
    type X: Sync;

    fn foo(&self, x: Self::X) -> i32 {
        todo!()
    }
}
"#,
        );
    }

    #[test]
    fn test_gen_blanket_cfg_attrs() {
        check_assist(
            generate_blanket_trait_impl,
            r#"
#[cfg(test)]
trait $0Foo {
    fn foo(&self, x: i32) -> i32;

    fn print_foo(&self) {
        println!("{}", self.foo());
    }
}
"#,
            r#"
#[cfg(test)]
trait Foo {
    fn foo(&self, x: i32) -> i32;

    fn print_foo(&self) {
        println!("{}", self.foo());
    }
}

#[cfg(test)]
impl<T: ?Sized> Foo for $0T {
    fn foo(&self, x: i32) -> i32 {
        todo!()
    }
}
"#,
        );
        check_assist(
            generate_blanket_trait_impl,
            r#"
#[cfg(test)]
trait $0Foo {
    /// ...
    #[cfg(test)]
    fn foo(&self, x: i32) -> i32;

    fn print_foo(&self) {
        println!("{}", self.foo());
    }
}
"#,
            r#"
#[cfg(test)]
trait Foo {
    /// ...
    #[cfg(test)]
    fn foo(&self, x: i32) -> i32;

    fn print_foo(&self) {
        println!("{}", self.foo());
    }
}

#[cfg(test)]
impl<T: ?Sized> Foo for $0T {
    #[cfg(test)]
    fn foo(&self, x: i32) -> i32 {
        todo!()
    }
}
"#,
        );
    }

    #[test]
    fn test_gen_blanket_empty_trait() {
        check_assist(
            generate_blanket_trait_impl,
            r#"
trait $0Foo {}
"#,
            r#"
trait Foo {}

impl<T: ?Sized> Foo for $0T {}
"#,
        );
    }

    #[test]
    fn test_gen_blanket_empty_trait_with_quick_bounds() {
        check_assist(
            generate_blanket_trait_impl,
            r#"
trait $0Foo: Copy {}
"#,
            r#"
trait Foo: Copy {}

impl<T: Copy + ?Sized> Foo for $0T {}
"#,
        );
    }

    #[test]
    fn test_gen_blanket_empty_trait_with_where_clause() {
        check_assist(
            generate_blanket_trait_impl,
            r#"
trait $0Foo where Self: Copy {}
"#,
            r#"
trait Foo where Self: Copy {}

impl<T: ?Sized> Foo for $0T
where Self: Copy
{
}
"#,
        );
    }

    #[test]
    fn test_gen_blanket_empty_trait_with_where_clause_comma() {
        check_assist(
            generate_blanket_trait_impl,
            r#"
trait $0Foo where Self: Copy, {}
"#,
            r#"
trait Foo where Self: Copy, {}

impl<T: ?Sized> Foo for $0T
where Self: Copy,
{
}
"#,
        );
    }

    #[test]
    fn test_gen_blanket_empty_trait_with_where_clause_newline() {
        check_assist(
            generate_blanket_trait_impl,
            r#"
trait $0Foo
where Self: Copy
{}
"#,
            r#"
trait Foo
where Self: Copy
{}

impl<T: ?Sized> Foo for $0T
where Self: Copy
{
}
"#,
        );
    }

    #[test]
    fn test_gen_blanket_empty_trait_with_where_clause_newline_newline() {
        check_assist(
            generate_blanket_trait_impl,
            r#"
trait $0Foo
where
    Self: Copy
{}
"#,
            r#"
trait Foo
where
    Self: Copy
{}

impl<T: ?Sized> Foo for $0T
where
    Self: Copy
{
}
"#,
        );
    }

    #[test]
    fn test_gen_blanket_empty_trait_with_where_clause_newline_newline_comma() {
        check_assist(
            generate_blanket_trait_impl,
            r#"
trait $0Foo
where
    Self: Copy,
{}
"#,
            r#"
trait Foo
where
    Self: Copy,
{}

impl<T: ?Sized> Foo for $0T
where
    Self: Copy,
{
}
"#,
        );
    }

    #[test]
    fn test_gen_blanket_empty_trait_with_multiple_where_clause() {
        check_assist(
            generate_blanket_trait_impl,
            r#"
trait $0Foo
where
    Self: Copy,
    (): Into<Self>,
{}
"#,
            r#"
trait Foo
where
    Self: Copy,
    (): Into<Self>,
{}

impl<T: ?Sized> Foo for $0T
where
    Self: Copy,
    (): Into<Self>,
{
}
"#,
        );
    }

    #[test]
    fn test_gen_blanket_empty_trait_with_multiple_bounds_where_clause() {
        check_assist(
            generate_blanket_trait_impl,
            r#"
trait $0Foo
where
    Self: Copy + Sync,
    (): Into<Self>,
{}
"#,
            r#"
trait Foo
where
    Self: Copy + Sync,
    (): Into<Self>,
{}

impl<T: ?Sized> Foo for $0T
where
    Self: Copy + Sync,
    (): Into<Self>,
{
}
"#,
        );
    }

    #[test]
    fn test_gen_blanket_empty_generate() {
        check_assist(
            generate_blanket_trait_impl,
            r#"
trait $0Foo {
    fn foo(&self) {}
}
"#,
            r#"
trait Foo {
    fn foo(&self) {}
}

impl<T: ?Sized> Foo for $0T {}
"#,
        );
    }

    #[test]
    fn test_gen_blanket_trait_with_doc() {
        check_assist(
            generate_blanket_trait_impl,
            r#"
/// some docs
trait $0Foo {}
"#,
            r#"
/// some docs
trait Foo {}

impl<T: ?Sized> Foo for $0T {}
"#,
        );
    }

    #[test]
    fn test_gen_blanket_multiple_method() {
        check_assist(
            generate_blanket_trait_impl,
            r#"
trait $0Foo {
    fn foo(&self);
    fn bar(&self);
}
"#,
            r#"
trait Foo {
    fn foo(&self);
    fn bar(&self);
}

impl<T: ?Sized> Foo for $0T {
    fn foo(&self) {
        todo!()
    }

    fn bar(&self) {
        todo!()
    }
}
"#,
        );
    }

    #[test]
    fn test_gen_blanket_method_with_generic() {
        check_assist(
            generate_blanket_trait_impl,
            r#"
trait $0Foo {
    fn foo<T>(&self, value: i32) -> i32;
    fn bar<T>(&self, value: i32) -> i32 { todo!() }
}
"#,
            r#"
trait Foo {
    fn foo<T>(&self, value: i32) -> i32;
    fn bar<T>(&self, value: i32) -> i32 { todo!() }
}

impl<T: ?Sized> Foo for $0T {
    fn foo<T>(&self, value: i32) -> i32 {
        todo!()
    }
}
"#,
        );
    }

    #[test]
    fn test_gen_blanket_method_with_lifetimes() {
        check_assist(
            generate_blanket_trait_impl,
            r#"
trait $0Foo<'a> {
    fn foo(&self) -> &'a str;
}
"#,
            r#"
trait Foo<'a> {
    fn foo(&self) -> &'a str;
}

impl<'a, T: ?Sized> Foo<'a> for $0T {
    fn foo(&self) -> &'a str {
        todo!()
    }
}
"#,
        );
    }

    #[test]
    fn test_gen_blanket_method_with_lifetime_bounds() {
        check_assist(
            generate_blanket_trait_impl,
            r#"
trait $0Foo<'a: 'static> {
    fn foo(&self) -> &'a str;
}
"#,
            r#"
trait Foo<'a: 'static> {
    fn foo(&self) -> &'a str;
}

impl<'a: 'static, T: ?Sized> Foo<'a> for $0T {
    fn foo(&self) -> &'a str {
        todo!()
    }
}
"#,
        );
    }

    #[test]
    fn test_gen_blanket_method_with_lifetime_quick_bounds() {
        check_assist(
            generate_blanket_trait_impl,
            r#"
trait $0Foo<'a>: 'a {
    fn foo(&self) -> &'a str;
}
"#,
            r#"
trait Foo<'a>: 'a {
    fn foo(&self) -> &'a str;
}

impl<'a, T: 'a + ?Sized> Foo<'a> for $0T {
    fn foo(&self) -> &'a str {
        todo!()
    }
}
"#,
        );
    }

    #[test]
    fn test_gen_blanket_method_with_multiple_lifetimes() {
        check_assist(
            generate_blanket_trait_impl,
            r#"
trait $0Foo<'a, 'b> {
    fn foo(&self) -> &'a &'b str;
}
"#,
            r#"
trait Foo<'a, 'b> {
    fn foo(&self) -> &'a &'b str;
}

impl<'a, 'b, T: ?Sized> Foo<'a, 'b> for $0T {
    fn foo(&self) -> &'a &'b str {
        todo!()
    }
}
"#,
        );
    }

    #[test]
    fn test_gen_blanket_method_with_lifetime_bounds_at_where_clause() {
        check_assist(
            generate_blanket_trait_impl,
            r#"
trait $0Foo<'a>
where 'a: 'static,
{
    fn foo(&self) -> &'a str;
}
"#,
            r#"
trait Foo<'a>
where 'a: 'static,
{
    fn foo(&self) -> &'a str;
}

impl<'a, T: ?Sized> Foo<'a> for $0T
where 'a: 'static,
{
    fn foo(&self) -> &'a str {
        todo!()
    }
}
"#,
        );
    }

    #[test]
    fn test_gen_blanket_not_on_name() {
        check_assist_not_applicable(
            generate_blanket_trait_impl,
            r#"
trait Foo<T: Send>: $0ToOwned
where
    Self::Owned: Default,
{
    fn foo(&self) -> T;

    fn print_foo(&self) {
        println!("{}", self.foo());
    }
}
"#,
        );

        check_assist_not_applicable(
            generate_blanket_trait_impl,
            r#"
trait Foo<T: Send>: ToOwned
where
    Self::Owned: Default,
{
    $0fn foo(&self) -> T;

    fn print_foo(&self) {
        println!("{}", self.foo());
    }
}
"#,
        );

        check_assist_not_applicable(
            generate_blanket_trait_impl,
            r#"
trait Foo<T: Send>: ToOwned
where
    Self::Owned: Default,
{
    fn $0foo(&self) -> T;

    fn print_foo(&self) {
        println!("{}", self.foo());
    }
}
"#,
        );
    }

    #[test]
    fn test_gen_blanket_existing_impl() {
        cov_mark::check!(existing_any_impl);
        check_assist_not_applicable(
            generate_blanket_trait_impl,
            r#"
trait $0Foo: Default {
    fn foo(&self) -> Self;
}
impl Foo for () {}
"#,
        );
    }

    #[test]
    fn test_gen_blanket_existing_other_impl() {
        check_assist(
            generate_blanket_trait_impl,
            r#"
trait $0Foo: Default {
    fn foo(&self) -> Self;
}
trait Bar: Default {
    fn bar(&self) -> Self;
}
impl Bar for () {}
"#,
            r#"
trait Foo: Default {
    fn foo(&self) -> Self;
}

impl<T: Default> Foo for $0T {
    fn foo(&self) -> Self {
        todo!()
    }
}
trait Bar: Default {
    fn bar(&self) -> Self;
}
impl Bar for () {}
"#,
        );
    }

    #[test]
    fn test_gen_blanket_apply_on_other_impl_block() {
        check_assist(
            generate_blanket_trait_impl,
            r#"
trait $0Foo {
    fn foo(&self) -> i32;

    fn print_foo(&self) {
        println!("{}", self.foo());
    }
}

trait Bar {}
impl Bar for i32 {}
"#,
            r#"
trait Foo {
    fn foo(&self) -> i32;

    fn print_foo(&self) {
        println!("{}", self.foo());
    }
}

impl<T: ?Sized> Foo for $0T {
    fn foo(&self) -> i32 {
        todo!()
    }
}

trait Bar {}
impl Bar for i32 {}
"#,
        );
    }

    #[test]
    fn test_gen_blanket_apply_on_other_blanket_impl_block() {
        check_assist(
            generate_blanket_trait_impl,
            r#"
trait $0Foo {
    fn foo(&self) -> i32;

    fn print_foo(&self) {
        println!("{}", self.foo());
    }
}

trait Bar {}
impl<T> Bar for T {}
"#,
            r#"
trait Foo {
    fn foo(&self) -> i32;

    fn print_foo(&self) {
        println!("{}", self.foo());
    }
}

impl<T: ?Sized> Foo for $0T {
    fn foo(&self) -> i32 {
        todo!()
    }
}

trait Bar {}
impl<T> Bar for T {}
"#,
        );
    }
}
