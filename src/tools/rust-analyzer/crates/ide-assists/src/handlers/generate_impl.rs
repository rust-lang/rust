use syntax::{
    ast::{self, edit_in_place::Indent, make, AstNode, HasName},
    ted,
};

use crate::{utils, AssistContext, AssistId, AssistKind, Assists};

fn insert_impl(impl_: ast::Impl, nominal: &ast::Adt) {
    let indent = nominal.indent_level();
    ted::insert_all_raw(
        ted::Position::after(nominal.syntax()),
        vec![
            // Add a blank line after the ADT, and indentation for the impl to match the ADT
            make::tokens::whitespace(&format!("\n\n{indent}")).into(),
            impl_.syntax().clone().into(),
        ],
    );
}

// Assist: generate_impl
//
// Adds a new inherent impl for a type.
//
// ```
// struct Ctx$0<T: Clone> {
//     data: T,
// }
// ```
// ->
// ```
// struct Ctx<T: Clone> {
//     data: T,
// }
//
// impl<T: Clone> Ctx<T> {$0}
// ```
pub(crate) fn generate_impl(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let nominal = ctx.find_node_at_offset::<ast::Adt>()?;
    let name = nominal.name()?;
    let target = nominal.syntax().text_range();

    if ctx.find_node_at_offset::<ast::RecordFieldList>().is_some() {
        return None;
    }

    acc.add(
        AssistId("generate_impl", AssistKind::Generate),
        format!("Generate impl for `{name}`"),
        target,
        |edit| {
            // Generate the impl
            let impl_ = utils::generate_impl(&nominal);

            // Add a tabstop after the left curly brace
            if let Some(cap) = ctx.config.snippet_cap {
                if let Some(l_curly) = impl_.assoc_item_list().and_then(|it| it.l_curly_token()) {
                    edit.add_tabstop_after_token(cap, l_curly);
                }
            }

            insert_impl(impl_, &edit.make_mut(nominal));
        },
    )
}

// Assist: generate_trait_impl
//
// Adds a new trait impl for a type.
//
// ```
// struct $0Ctx<T: Clone> {
//     data: T,
// }
// ```
// ->
// ```
// struct Ctx<T: Clone> {
//     data: T,
// }
//
// impl<T: Clone> ${1:_} for Ctx<T> {$0}
// ```
pub(crate) fn generate_trait_impl(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let nominal = ctx.find_node_at_offset::<ast::Adt>()?;
    let name = nominal.name()?;
    let target = nominal.syntax().text_range();

    if ctx.find_node_at_offset::<ast::RecordFieldList>().is_some() {
        return None;
    }

    acc.add(
        AssistId("generate_trait_impl", AssistKind::Generate),
        format!("Generate trait impl for `{name}`"),
        target,
        |edit| {
            // Generate the impl
            let impl_ = utils::generate_trait_impl_intransitive(&nominal, make::ty_placeholder());

            // Make the trait type a placeholder snippet
            if let Some(cap) = ctx.config.snippet_cap {
                if let Some(trait_) = impl_.trait_() {
                    edit.add_placeholder_snippet(cap, trait_);
                }

                if let Some(l_curly) = impl_.assoc_item_list().and_then(|it| it.l_curly_token()) {
                    edit.add_tabstop_after_token(cap, l_curly);
                }
            }

            insert_impl(impl_, &edit.make_mut(nominal));
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_target};

    use super::*;

    #[test]
    fn test_add_impl() {
        check_assist(
            generate_impl,
            r#"
                struct Foo$0 {}
            "#,
            r#"
                struct Foo {}

                impl Foo {$0}
            "#,
        );
    }

    #[test]
    fn test_add_impl_with_generics() {
        check_assist(
            generate_impl,
            r#"
                struct Foo$0<T: Clone> {}
            "#,
            r#"
                struct Foo<T: Clone> {}

                impl<T: Clone> Foo<T> {$0}
            "#,
        );
    }

    #[test]
    fn test_add_impl_with_generics_and_lifetime_parameters() {
        check_assist(
            generate_impl,
            r#"
                struct Foo<'a, T: Foo<'a>>$0 {}
            "#,
            r#"
                struct Foo<'a, T: Foo<'a>> {}

                impl<'a, T: Foo<'a>> Foo<'a, T> {$0}
            "#,
        );
    }

    #[test]
    fn test_add_impl_with_attributes() {
        check_assist(
            generate_impl,
            r#"
                #[cfg(feature = "foo")]
                struct Foo<'a, T: Foo$0<'a>> {}
            "#,
            r#"
                #[cfg(feature = "foo")]
                struct Foo<'a, T: Foo<'a>> {}

                #[cfg(feature = "foo")]
                impl<'a, T: Foo<'a>> Foo<'a, T> {$0}
            "#,
        );
    }

    #[test]
    fn test_add_impl_with_default_generic() {
        check_assist(
            generate_impl,
            r#"
                struct Defaulted$0<T = i32> {}
            "#,
            r#"
                struct Defaulted<T = i32> {}

                impl<T> Defaulted<T> {$0}
            "#,
        );
    }

    #[test]
    fn test_add_impl_with_constrained_default_generic() {
        check_assist(
            generate_impl,
            r#"
                struct Defaulted$0<'a, 'b: 'a, T: Debug + Clone + 'a + 'b = String, const S: usize> {}
            "#,
            r#"
                struct Defaulted<'a, 'b: 'a, T: Debug + Clone + 'a + 'b = String, const S: usize> {}

                impl<'a, 'b: 'a, T: Debug + Clone + 'a + 'b, const S: usize> Defaulted<'a, 'b, T, S> {$0}
            "#,
        );
    }

    #[test]
    fn test_add_impl_with_const_defaulted_generic() {
        check_assist(
            generate_impl,
            r#"
                struct Defaulted$0<const N: i32 = 0> {}
            "#,
            r#"
                struct Defaulted<const N: i32 = 0> {}

                impl<const N: i32> Defaulted<N> {$0}
            "#,
        );
    }

    #[test]
    fn test_add_impl_with_trait_constraint() {
        check_assist(
            generate_impl,
            r#"
                pub trait Trait {}
                struct Struct$0<T>
                where
                    T: Trait,
                {
                    inner: T,
                }
            "#,
            r#"
                pub trait Trait {}
                struct Struct<T>
                where
                    T: Trait,
                {
                    inner: T,
                }

                impl<T> Struct<T>
                where
                    T: Trait,
                {$0
                }
            "#,
        );
    }

    #[test]
    fn add_impl_target() {
        check_assist_target(
            generate_impl,
            r#"
                struct SomeThingIrrelevant;
                /// Has a lifetime parameter
                struct Foo$0<'a, T: Foo<'a>> {}
                struct EvenMoreIrrelevant;
            "#,
            "/// Has a lifetime parameter\nstruct Foo<'a, T: Foo<'a>> {}",
        );
    }

    #[test]
    fn test_add_trait_impl() {
        check_assist(
            generate_trait_impl,
            r#"
                struct Foo$0 {}
            "#,
            r#"
                struct Foo {}

                impl ${1:_} for Foo {$0}
            "#,
        );
    }

    #[test]
    fn test_add_trait_impl_with_generics() {
        check_assist(
            generate_trait_impl,
            r#"
                struct Foo$0<T: Clone> {}
            "#,
            r#"
                struct Foo<T: Clone> {}

                impl<T: Clone> ${1:_} for Foo<T> {$0}
            "#,
        );
    }

    #[test]
    fn test_add_trait_impl_with_generics_and_lifetime_parameters() {
        check_assist(
            generate_trait_impl,
            r#"
                struct Foo<'a, T: Foo<'a>>$0 {}
            "#,
            r#"
                struct Foo<'a, T: Foo<'a>> {}

                impl<'a, T: Foo<'a>> ${1:_} for Foo<'a, T> {$0}
            "#,
        );
    }

    #[test]
    fn test_add_trait_impl_with_attributes() {
        check_assist(
            generate_trait_impl,
            r#"
                #[cfg(feature = "foo")]
                struct Foo<'a, T: Foo$0<'a>> {}
            "#,
            r#"
                #[cfg(feature = "foo")]
                struct Foo<'a, T: Foo<'a>> {}

                #[cfg(feature = "foo")]
                impl<'a, T: Foo<'a>> ${1:_} for Foo<'a, T> {$0}
            "#,
        );
    }

    #[test]
    fn test_add_trait_impl_with_default_generic() {
        check_assist(
            generate_trait_impl,
            r#"
                struct Defaulted$0<T = i32> {}
            "#,
            r#"
                struct Defaulted<T = i32> {}

                impl<T> ${1:_} for Defaulted<T> {$0}
            "#,
        );
    }

    #[test]
    fn test_add_trait_impl_with_constrained_default_generic() {
        check_assist(
            generate_trait_impl,
            r#"
                struct Defaulted$0<'a, 'b: 'a, T: Debug + Clone + 'a + 'b = String, const S: usize> {}
            "#,
            r#"
                struct Defaulted<'a, 'b: 'a, T: Debug + Clone + 'a + 'b = String, const S: usize> {}

                impl<'a, 'b: 'a, T: Debug + Clone + 'a + 'b, const S: usize> ${1:_} for Defaulted<'a, 'b, T, S> {$0}
            "#,
        );
    }

    #[test]
    fn test_add_trait_impl_with_const_defaulted_generic() {
        check_assist(
            generate_trait_impl,
            r#"
                struct Defaulted$0<const N: i32 = 0> {}
            "#,
            r#"
                struct Defaulted<const N: i32 = 0> {}

                impl<const N: i32> ${1:_} for Defaulted<N> {$0}
            "#,
        );
    }

    #[test]
    fn test_add_trait_impl_with_trait_constraint() {
        check_assist(
            generate_trait_impl,
            r#"
                pub trait Trait {}
                struct Struct$0<T>
                where
                    T: Trait,
                {
                    inner: T,
                }
            "#,
            r#"
                pub trait Trait {}
                struct Struct<T>
                where
                    T: Trait,
                {
                    inner: T,
                }

                impl<T> ${1:_} for Struct<T>
                where
                    T: Trait,
                {$0
                }
            "#,
        );
    }

    #[test]
    fn add_trait_impl_target() {
        check_assist_target(
            generate_trait_impl,
            r#"
                struct SomeThingIrrelevant;
                /// Has a lifetime parameter
                struct Foo$0<'a, T: Foo<'a>> {}
                struct EvenMoreIrrelevant;
            "#,
            "/// Has a lifetime parameter\nstruct Foo<'a, T: Foo<'a>> {}",
        );
    }

    #[test]
    fn add_impl_with_indent() {
        check_assist(
            generate_impl,
            r#"
                mod foo {
                    struct Bar$0 {}
                }
            "#,
            r#"
                mod foo {
                    struct Bar {}

                    impl Bar {$0}
                }
            "#,
        );
    }

    #[test]
    fn add_impl_with_multiple_indent() {
        check_assist(
            generate_impl,
            r#"
                mod foo {
                    fn bar() {
                        struct Baz$0 {}
                    }
                }
            "#,
            r#"
                mod foo {
                    fn bar() {
                        struct Baz {}

                        impl Baz {$0}
                    }
                }
            "#,
        );
    }

    #[test]
    fn add_trait_impl_with_indent() {
        check_assist(
            generate_trait_impl,
            r#"
                mod foo {
                    struct Bar$0 {}
                }
            "#,
            r#"
                mod foo {
                    struct Bar {}

                    impl ${1:_} for Bar {$0}
                }
            "#,
        );
    }
}
