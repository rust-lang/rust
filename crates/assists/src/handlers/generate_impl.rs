use itertools::Itertools;
use stdx::format_to;
use syntax::{
    ast::{self, AstNode, AttrsOwner, GenericParamsOwner, NameOwner},
    SmolStr,
};

use crate::{AssistContext, AssistId, AssistKind, Assists};

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
    let nominal = ctx.find_node_at_offset::<ast::AdtDef>()?;
    let name = nominal.name()?;
    let target = nominal.syntax().text_range();

    acc.add(
        AssistId("generate_impl", AssistKind::Generate),
        format!("Generate impl for `{}`", name),
        target,
        |edit| {
            let type_params = nominal.generic_param_list();
            let start_offset = nominal.syntax().text_range().end();
            let mut buf = String::new();
            buf.push_str("\n\n");
            nominal
                .attrs()
                .filter(|attr| {
                    attr.as_simple_call().map(|(name, _arg)| name == "cfg").unwrap_or(false)
                })
                .for_each(|attr| buf.push_str(format!("{}\n", attr.to_string()).as_str()));

            buf.push_str("impl");
            if let Some(type_params) = &type_params {
                format_to!(buf, "{}", type_params.syntax());
            }
            buf.push_str(" ");
            buf.push_str(name.text());
            if let Some(type_params) = type_params {
                let lifetime_params = type_params
                    .lifetime_params()
                    .filter_map(|it| it.lifetime())
                    .map(|it| SmolStr::from(it.text()));
                let type_params = type_params
                    .type_params()
                    .filter_map(|it| it.name())
                    .map(|it| SmolStr::from(it.text()));

                let generic_params = lifetime_params.chain(type_params).format(", ");
                format_to!(buf, "<{}>", generic_params)
            }
            match ctx.config.snippet_cap {
                Some(cap) => {
                    buf.push_str(" {\n    $0\n}");
                    edit.insert_snippet(cap, start_offset, buf);
                }
                None => {
                    buf.push_str(" {\n}");
                    edit.insert(start_offset, buf);
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
