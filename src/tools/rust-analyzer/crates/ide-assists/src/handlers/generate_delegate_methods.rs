use hir::{HasCrate, HasVisibility};
use ide_db::{FxHashSet, path_transform::PathTransform};
use syntax::{
    ast::{
        self, AstNode, HasGenericParams, HasName, HasVisibility as _,
        edit::{AstNodeEdit, IndentLevel},
        make,
    },
    syntax_editor::Position,
};

use crate::{
    AssistContext, AssistId, AssistKind, Assists, GroupLabel,
    utils::{convert_param_list_to_arg_list, find_struct_impl},
};

// Assist: generate_delegate_methods
//
// Generate delegate methods.
//
// ```
// struct Age(u8);
// impl Age {
//     fn age(&self) -> u8 {
//         self.0
//     }
// }
//
// struct Person {
//     ag$0e: Age,
// }
// ```
// ->
// ```
// struct Age(u8);
// impl Age {
//     fn age(&self) -> u8 {
//         self.0
//     }
// }
//
// struct Person {
//     age: Age,
// }
//
// impl Person {
//     $0fn age(&self) -> u8 {
//         self.age.age()
//     }
// }
// ```
pub(crate) fn generate_delegate_methods(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    if !ctx.config.code_action_grouping {
        return None;
    }

    let strukt = ctx.find_node_at_offset::<ast::Struct>()?;
    let strukt_name = strukt.name()?;
    let current_module = ctx.sema.scope(strukt.syntax())?.module();
    let current_edition = current_module.krate().edition(ctx.db());

    let (field_name, field_ty, target) = match ctx.find_node_at_offset::<ast::RecordField>() {
        Some(field) => {
            let field_name = field.name()?;
            let field_ty = field.ty()?;
            (field_name.to_string(), field_ty, field.syntax().text_range())
        }
        None => {
            let field = ctx.find_node_at_offset::<ast::TupleField>()?;
            let field_list = ctx.find_node_at_offset::<ast::TupleFieldList>()?;
            let field_list_index = field_list.fields().position(|it| it == field)?;
            let field_ty = field.ty()?;
            (field_list_index.to_string(), field_ty, field.syntax().text_range())
        }
    };

    let sema_field_ty = ctx.sema.resolve_type(&field_ty)?;
    let mut methods = vec![];
    let mut seen_names = FxHashSet::default();

    for ty in sema_field_ty.autoderef(ctx.db()) {
        let krate = ty.krate(ctx.db());
        ty.iterate_assoc_items(ctx.db(), krate, |item| {
            if let hir::AssocItem::Function(f) = item {
                let name = f.name(ctx.db());
                if f.self_param(ctx.db()).is_some()
                    && f.is_visible_from(ctx.db(), current_module)
                    && seen_names.insert(name.clone())
                {
                    methods.push((name, f))
                }
            }
            Option::<()>::None
        });
    }
    methods.sort_by(|(a, _), (b, _)| a.cmp(b));
    for (index, (name, method)) in methods.into_iter().enumerate() {
        let adt = ast::Adt::Struct(strukt.clone());
        let name = name.display(ctx.db(), current_edition).to_string();
        // if `find_struct_impl` returns None, that means that a function named `name` already exists.
        let Some(impl_def) = find_struct_impl(ctx, &adt, std::slice::from_ref(&name)) else {
            continue;
        };
        let field = make::ext::field_from_idents(["self", &field_name])?;

        acc.add_group(
            &GroupLabel("Generate delegate methods…".to_owned()),
            AssistId("generate_delegate_methods", AssistKind::Generate, Some(index)),
            format!("Generate delegate for `{field_name}.{name}()`",),
            target,
            |edit| {
                // Create the function
                let method_source = match ctx.sema.source(method) {
                    Some(source) => {
                        let v = source.value.clone_for_update();
                        let source_scope = ctx.sema.scope(v.syntax());
                        let target_scope = ctx.sema.scope(strukt.syntax());
                        if let (Some(s), Some(t)) = (source_scope, target_scope) {
                            ast::Fn::cast(
                                PathTransform::generic_transformation(&t, &s).apply(v.syntax()),
                            )
                            .unwrap_or(v)
                        } else {
                            v
                        }
                    }
                    None => return,
                };

                let vis = method_source.visibility();
                let is_async = method_source.async_token().is_some();
                let is_const = method_source.const_token().is_some();
                let is_unsafe = method_source.unsafe_token().is_some();
                let is_gen = method_source.gen_token().is_some();

                let fn_name = make::name(&name);

                let type_params = method_source.generic_param_list();
                let where_clause = method_source.where_clause();
                let params =
                    method_source.param_list().unwrap_or_else(|| make::param_list(None, []));

                // compute the `body`
                let arg_list = method_source
                    .param_list()
                    .map(convert_param_list_to_arg_list)
                    .unwrap_or_else(|| make::arg_list([]));

                let tail_expr =
                    make::expr_method_call(field, make::name_ref(&name), arg_list).into();
                let tail_expr_finished =
                    if is_async { make::expr_await(tail_expr) } else { tail_expr };
                let body = make::block_expr([], Some(tail_expr_finished));

                let ret_type = method_source.ret_type();

                let f = make::fn_(
                    vis,
                    fn_name,
                    type_params,
                    where_clause,
                    params,
                    body,
                    ret_type,
                    is_async,
                    is_const,
                    is_unsafe,
                    is_gen,
                )
                .indent(IndentLevel(1));
                let item = ast::AssocItem::Fn(f.clone());

                let mut editor = edit.make_editor(strukt.syntax());
                let fn_: Option<ast::AssocItem> = match impl_def {
                    Some(impl_def) => match impl_def.assoc_item_list() {
                        Some(assoc_item_list) => {
                            let item = item.indent(IndentLevel::from_node(impl_def.syntax()));
                            assoc_item_list.add_items(&mut editor, vec![item.clone()]);
                            Some(item)
                        }
                        None => {
                            let assoc_item_list = make::assoc_item_list(Some(vec![item]));
                            editor.insert(
                                Position::last_child_of(impl_def.syntax()),
                                assoc_item_list.syntax(),
                            );
                            assoc_item_list.assoc_items().next()
                        }
                    },
                    None => {
                        let name = &strukt_name.to_string();
                        let ty_params = strukt.generic_param_list();
                        let ty_args = ty_params.as_ref().map(|it| it.to_generic_args());
                        let where_clause = strukt.where_clause();
                        let assoc_item_list = make::assoc_item_list(Some(vec![item]));

                        let impl_def = make::impl_(
                            ty_params,
                            ty_args,
                            make::ty_path(make::ext::ident_path(name)),
                            where_clause,
                            Some(assoc_item_list),
                        )
                        .clone_for_update();

                        // Fixup impl_def indentation
                        let indent = strukt.indent_level();
                        let impl_def = impl_def.indent(indent);

                        // Insert the impl block.
                        let strukt = edit.make_mut(strukt.clone());
                        editor.insert_all(
                            Position::after(strukt.syntax()),
                            vec![
                                make::tokens::whitespace(&format!("\n\n{indent}")).into(),
                                impl_def.syntax().clone().into(),
                            ],
                        );
                        impl_def.assoc_item_list().and_then(|list| list.assoc_items().next())
                    }
                };

                if let Some(cap) = ctx.config.snippet_cap
                    && let Some(fn_) = fn_
                {
                    let tabstop = edit.make_tabstop_before(cap);
                    editor.add_annotation(fn_.syntax(), tabstop);
                }
                edit.add_file_edits(ctx.vfs_file_id(), editor);
            },
        )?;
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use crate::tests::{
        check_assist, check_assist_not_applicable, check_assist_not_applicable_no_grouping,
    };

    use super::*;

    #[test]
    fn test_generate_delegate_create_impl_block() {
        check_assist(
            generate_delegate_methods,
            r#"
struct Age(u8);
impl Age {
    fn age(&self) -> u8 {
        self.0
    }
}

struct Person {
    ag$0e: Age,
}"#,
            r#"
struct Age(u8);
impl Age {
    fn age(&self) -> u8 {
        self.0
    }
}

struct Person {
    age: Age,
}

impl Person {
    $0fn age(&self) -> u8 {
        self.age.age()
    }
}"#,
        );
    }

    #[test]
    fn test_generate_delegate_create_impl_block_match_indent() {
        check_assist(
            generate_delegate_methods,
            r#"
mod indent {
    struct Age(u8);
    impl Age {
        fn age(&self) -> u8 {
            self.0
        }
    }

    struct Person {
        ag$0e: Age,
    }
}"#,
            r#"
mod indent {
    struct Age(u8);
    impl Age {
        fn age(&self) -> u8 {
            self.0
        }
    }

    struct Person {
        age: Age,
    }

    impl Person {
        $0fn age(&self) -> u8 {
            self.age.age()
        }
    }
}"#,
        );
    }

    #[test]
    fn test_generate_delegate_update_impl_block() {
        check_assist(
            generate_delegate_methods,
            r#"
struct Age(u8);
impl Age {
    fn age(&self) -> u8 {
        self.0
    }
}

struct Person {
    ag$0e: Age,
}

impl Person {}"#,
            r#"
struct Age(u8);
impl Age {
    fn age(&self) -> u8 {
        self.0
    }
}

struct Person {
    age: Age,
}

impl Person {
    $0fn age(&self) -> u8 {
        self.age.age()
    }
}"#,
        );
    }

    #[test]
    fn test_generate_delegate_update_impl_block_match_indent() {
        check_assist(
            generate_delegate_methods,
            r#"
mod indent {
    struct Age(u8);
    impl Age {
        fn age(&self) -> u8 {
            self.0
        }
    }

    struct Person {
        ag$0e: Age,
    }

    impl Person {}
}"#,
            r#"
mod indent {
    struct Age(u8);
    impl Age {
        fn age(&self) -> u8 {
            self.0
        }
    }

    struct Person {
        age: Age,
    }

    impl Person {
        $0fn age(&self) -> u8 {
            self.age.age()
        }
    }
}"#,
        );
    }

    #[test]
    fn test_generate_delegate_tuple_struct() {
        check_assist(
            generate_delegate_methods,
            r#"
struct Age(u8);
impl Age {
    fn age(&self) -> u8 {
        self.0
    }
}

struct Person(A$0ge);"#,
            r#"
struct Age(u8);
impl Age {
    fn age(&self) -> u8 {
        self.0
    }
}

struct Person(Age);

impl Person {
    $0fn age(&self) -> u8 {
        self.0.age()
    }
}"#,
        );
    }

    #[test]
    fn test_generate_delegate_enable_all_attributes() {
        check_assist(
            generate_delegate_methods,
            r#"
struct Age<T>(T);
impl<T> Age<T> {
    pub(crate) async fn age<J, 'a>(&'a mut self, ty: T, arg: J) -> T {
        self.0
    }
}

struct Person<T> {
    ag$0e: Age<T>,
}"#,
            r#"
struct Age<T>(T);
impl<T> Age<T> {
    pub(crate) async fn age<J, 'a>(&'a mut self, ty: T, arg: J) -> T {
        self.0
    }
}

struct Person<T> {
    age: Age<T>,
}

impl<T> Person<T> {
    $0pub(crate) async fn age<J, 'a>(&'a mut self, ty: T, arg: J) -> T {
        self.age.age(ty, arg).await
    }
}"#,
        );
    }

    #[test]
    fn test_generates_delegate_autoderef() {
        check_assist(
            generate_delegate_methods,
            r#"
//- minicore: deref
struct Age(u8);
impl Age {
    fn age(&self) -> u8 {
        self.0
    }
}
struct AgeDeref(Age);
impl core::ops::Deref for AgeDeref { type Target = Age; }
struct Person {
    ag$0e: AgeDeref,
}
impl Person {}"#,
            r#"
struct Age(u8);
impl Age {
    fn age(&self) -> u8 {
        self.0
    }
}
struct AgeDeref(Age);
impl core::ops::Deref for AgeDeref { type Target = Age; }
struct Person {
    age: AgeDeref,
}
impl Person {
    $0fn age(&self) -> u8 {
        self.age.age()
    }
}"#,
        );
    }

    #[test]
    fn test_preserve_where_clause() {
        check_assist(
            generate_delegate_methods,
            r#"
struct Inner<T>(T);
impl<T> Inner<T> {
    fn get(&self) -> T
    where
        T: Copy,
        T: PartialEq,
    {
        self.0
    }
}

struct Struct<T> {
    $0field: Inner<T>,
}
"#,
            r#"
struct Inner<T>(T);
impl<T> Inner<T> {
    fn get(&self) -> T
    where
        T: Copy,
        T: PartialEq,
    {
        self.0
    }
}

struct Struct<T> {
    field: Inner<T>,
}

impl<T> Struct<T> {
    $0fn get(&self) -> T where
            T: Copy,
            T: PartialEq, {
        self.field.get()
    }
}
"#,
        );
    }

    #[test]
    fn test_fixes_basic_self_references() {
        check_assist(
            generate_delegate_methods,
            r#"
struct Foo {
    field: $0Bar,
}

struct Bar;

impl Bar {
    fn bar(&self, other: Self) -> Self {
        other
    }
}
"#,
            r#"
struct Foo {
    field: Bar,
}

impl Foo {
    $0fn bar(&self, other: Bar) -> Bar {
        self.field.bar(other)
    }
}

struct Bar;

impl Bar {
    fn bar(&self, other: Self) -> Self {
        other
    }
}
"#,
        );
    }

    #[test]
    fn test_fixes_nested_self_references() {
        check_assist(
            generate_delegate_methods,
            r#"
struct Foo {
    field: $0Bar,
}

struct Bar;

impl Bar {
    fn bar(&mut self, a: (Self, [Self; 4]), b: Vec<Self>) {}
}
"#,
            r#"
struct Foo {
    field: Bar,
}

impl Foo {
    $0fn bar(&mut self, a: (Bar, [Bar; 4]), b: Vec<Bar>) {
        self.field.bar(a, b)
    }
}

struct Bar;

impl Bar {
    fn bar(&mut self, a: (Self, [Self; 4]), b: Vec<Self>) {}
}
"#,
        );
    }

    #[test]
    fn test_fixes_self_references_with_lifetimes_and_generics() {
        check_assist(
            generate_delegate_methods,
            r#"
struct Foo<'a, T> {
    $0field: Bar<'a, T>,
}

struct Bar<'a, T>(&'a T);

impl<'a, T> Bar<'a, T> {
    fn bar(self, mut b: Vec<&'a Self>) -> &'a Self {
        b.pop().unwrap()
    }
}
"#,
            r#"
struct Foo<'a, T> {
    field: Bar<'a, T>,
}

impl<'a, T> Foo<'a, T> {
    $0fn bar(self, mut b: Vec<&'a Bar<'a, T>>) -> &'a Bar<'a, T> {
        self.field.bar(b)
    }
}

struct Bar<'a, T>(&'a T);

impl<'a, T> Bar<'a, T> {
    fn bar(self, mut b: Vec<&'a Self>) -> &'a Self {
        b.pop().unwrap()
    }
}
"#,
        );
    }

    #[test]
    fn test_fixes_self_references_across_macros() {
        check_assist(
            generate_delegate_methods,
            r#"
//- /bar.rs
macro_rules! test_method {
    () => {
        pub fn test(self, b: Bar) -> Self {
            self
        }
    };
}

pub struct Bar;

impl Bar {
    test_method!();
}

//- /main.rs
mod bar;

struct Foo {
    $0bar: bar::Bar,
}
"#,
            r#"
mod bar;

struct Foo {
    bar: bar::Bar,
}

impl Foo {
    $0pub fn test(self,b:bar::Bar) ->bar::Bar {
        self.bar.test(b)
    }
}
"#,
        );
    }

    #[test]
    fn test_generate_delegate_visibility() {
        check_assist_not_applicable(
            generate_delegate_methods,
            r#"
mod m {
    pub struct Age(u8);
    impl Age {
        fn age(&self) -> u8 {
            self.0
        }
    }
}

struct Person {
    ag$0e: m::Age,
}"#,
        )
    }

    #[test]
    fn test_generate_not_eligible_if_fn_exists() {
        check_assist_not_applicable(
            generate_delegate_methods,
            r#"
struct Age(u8);
impl Age {
    fn age(&self) -> u8 {
        self.0
    }
}

struct Person {
    ag$0e: Age,
}
impl Person {
    fn age(&self) -> u8 { 0 }
}
"#,
        );
    }

    #[test]
    fn delegate_method_skipped_when_no_grouping() {
        check_assist_not_applicable_no_grouping(
            generate_delegate_methods,
            r#"
struct Age(u8);
impl Age {
    fn age(&self) -> u8 {
        self.0
    }
}
struct Person {
    ag$0e: Age,
}"#,
        );
    }
}
