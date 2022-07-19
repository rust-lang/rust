use syntax::ast::{self, AstNode, HasName};

use crate::{utils::generate_impl_text, AssistContext, AssistId, AssistKind, Assists};

// Assist: generate_impl
//
// Adds a new inherent impl for a type.
//
// ```
// struct Ctx<T: Clone> {
//     data: T,$0
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
pub(crate) fn generate_impl(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let nominal = ctx.find_node_at_offset::<ast::Adt>()?;
    let name = nominal.name()?;
    let target = nominal.syntax().text_range();

    acc.add(
        AssistId("generate_impl", AssistKind::Generate),
        format!("Generate impl for `{}`", name),
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

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_target};

    use super::*;

    #[test]
    fn test_add_impl() {
        check_assist(
            generate_impl,
            "struct Foo {$0}\n",
            "struct Foo {}\n\nimpl Foo {\n    $0\n}\n",
        );
        check_assist(
            generate_impl,
            "struct Foo<T: Clone> {$0}",
            "struct Foo<T: Clone> {}\n\nimpl<T: Clone> Foo<T> {\n    $0\n}",
        );
        check_assist(
            generate_impl,
            "struct Foo<'a, T: Foo<'a>> {$0}",
            "struct Foo<'a, T: Foo<'a>> {}\n\nimpl<'a, T: Foo<'a>> Foo<'a, T> {\n    $0\n}",
        );
        check_assist(
            generate_impl,
            r#"
            struct MyOwnArray<T, const S: usize> {}$0"#,
            r#"
            struct MyOwnArray<T, const S: usize> {}

            impl<T, const S: usize> MyOwnArray<T, S> {
                $0
            }"#,
        );
        check_assist(
            generate_impl,
            r#"
            #[cfg(feature = "foo")]
            struct Foo<'a, T: Foo<'a>> {$0}"#,
            r#"
            #[cfg(feature = "foo")]
            struct Foo<'a, T: Foo<'a>> {}

            #[cfg(feature = "foo")]
            impl<'a, T: Foo<'a>> Foo<'a, T> {
                $0
            }"#,
        );

        check_assist(
            generate_impl,
            r#"
            #[cfg(not(feature = "foo"))]
            struct Foo<'a, T: Foo<'a>> {$0}"#,
            r#"
            #[cfg(not(feature = "foo"))]
            struct Foo<'a, T: Foo<'a>> {}

            #[cfg(not(feature = "foo"))]
            impl<'a, T: Foo<'a>> Foo<'a, T> {
                $0
            }"#,
        );

        check_assist(
            generate_impl,
            r#"
            struct Defaulted<T = i32> {}$0"#,
            r#"
            struct Defaulted<T = i32> {}

            impl<T> Defaulted<T> {
                $0
            }"#,
        );

        check_assist(
            generate_impl,
            r#"
            struct Defaulted<'a, 'b: 'a, T: Debug + Clone + 'a + 'b = String, const S: usize> {}$0"#,
            r#"
            struct Defaulted<'a, 'b: 'a, T: Debug + Clone + 'a + 'b = String, const S: usize> {}

            impl<'a, 'b: 'a, T: Debug + Clone + 'a + 'b, const S: usize> Defaulted<'a, 'b, T, S> {
                $0
            }"#,
        );

        check_assist(
            generate_impl,
            r#"pub trait Trait {}
struct Struct<T>$0
where
    T: Trait,
{
    inner: T,
}"#,
            r#"pub trait Trait {}
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
}"#,
        );
    }

    #[test]
    fn add_impl_target() {
        check_assist_target(
            generate_impl,
            "
struct SomeThingIrrelevant;
/// Has a lifetime parameter
struct Foo<'a, T: Foo<'a>> {$0}
struct EvenMoreIrrelevant;
",
            "/// Has a lifetime parameter
struct Foo<'a, T: Foo<'a>> {}",
        );
    }
}
