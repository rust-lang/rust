use syntax::ast::{self, AstNode, HasName};

use crate::{
    utils::{generate_impl_text, generate_trait_impl_text_intransitive},
    AssistContext, AssistId, AssistKind, Assists,
};

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
// impl<T: Clone> Ctx<T> {
//     $0
// }
// ```
pub(crate) fn generate_impl(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let nominal = ctx.find_node_at_offset::<ast::Adt>()?;
    let name = nominal.name()?;
    let target = nominal.syntax().text_range();

    if let Some(_) = ctx.find_node_at_offset::<ast::RecordFieldList>() {
        return None;
    }

    acc.add(
        AssistId("generate_impl", AssistKind::Generate),
        format!("Generate impl for `{name}`"),
        target,
        |edit| {
            let start_offset = nominal.syntax().text_range().end();
            match ctx.config.snippet_cap {
                Some(cap) => {
                    let snippet = generate_impl_text(&nominal, "    $0");
                    edit.insert_snippet(cap, start_offset, snippet);
                }
                None => {
                    let snippet = generate_impl_text(&nominal, "");
                    edit.insert(start_offset, snippet);
                }
            }
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
// impl<T: Clone> $0 for Ctx<T> {
//
// }
// ```
pub(crate) fn generate_trait_impl(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let nominal = ctx.find_node_at_offset::<ast::Adt>()?;
    let name = nominal.name()?;
    let target = nominal.syntax().text_range();

    if let Some(_) = ctx.find_node_at_offset::<ast::RecordFieldList>() {
        return None;
    }

    acc.add(
        AssistId("generate_trait_impl", AssistKind::Generate),
        format!("Generate trait impl for `{name}`"),
        target,
        |edit| {
            let start_offset = nominal.syntax().text_range().end();
            match ctx.config.snippet_cap {
                Some(cap) => {
                    let snippet = generate_trait_impl_text_intransitive(&nominal, "$0", "");
                    edit.insert_snippet(cap, start_offset, snippet);
                }
                None => {
                    let text = generate_trait_impl_text_intransitive(&nominal, "", "");
                    edit.insert(start_offset, text);
                }
            }
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

                impl Foo {
                    $0
                }
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

                impl<T: Clone> Foo<T> {
                    $0
                }
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

                impl<'a, T: Foo<'a>> Foo<'a, T> {
                    $0
                }
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
                impl<'a, T: Foo<'a>> Foo<'a, T> {
                    $0
                }
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

                impl<T> Defaulted<T> {
                    $0
                }
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

                impl<'a, 'b: 'a, T: Debug + Clone + 'a + 'b, const S: usize> Defaulted<'a, 'b, T, S> {
                    $0
                }
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

                impl<const N: i32> Defaulted<N> {
                    $0
                }
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
                {
                    $0
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

                impl $0 for Foo {

                }
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

                impl<T: Clone> $0 for Foo<T> {

                }
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

                impl<'a, T: Foo<'a>> $0 for Foo<'a, T> {

                }
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
                impl<'a, T: Foo<'a>> $0 for Foo<'a, T> {

                }
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

                impl<T> $0 for Defaulted<T> {

                }
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

                impl<'a, 'b: 'a, T: Debug + Clone + 'a + 'b, const S: usize> $0 for Defaulted<'a, 'b, T, S> {

                }
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

                impl<const N: i32> $0 for Defaulted<N> {

                }
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

                impl<T> $0 for Struct<T>
                where
                    T: Trait,
                {

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
}
