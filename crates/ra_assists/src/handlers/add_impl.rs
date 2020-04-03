use ra_syntax::{
    ast::{self, AstNode, AstToken, NameOwner, TypeParamsOwner},
    TextUnit,
};
use stdx::{format_to, SepBy};

use crate::{Assist, AssistCtx, AssistId};

// Assist: add_impl
//
// Adds a new inherent impl for a type.
//
// ```
// struct Ctx<T: Clone> {
//      data: T,<|>
// }
// ```
// ->
// ```
// struct Ctx<T: Clone> {
//      data: T,
// }
//
// impl<T: Clone> Ctx<T> {
//
// }
// ```
pub(crate) fn add_impl(ctx: AssistCtx) -> Option<Assist> {
    let nominal = ctx.find_node_at_offset::<ast::NominalDef>()?;
    let name = nominal.name()?;
    ctx.add_assist(AssistId("add_impl"), format!("Implement {}", name.text().as_str()), |edit| {
        edit.target(nominal.syntax().text_range());
        let type_params = nominal.type_param_list();
        let start_offset = nominal.syntax().text_range().end();
        let mut buf = String::new();
        buf.push_str("\n\nimpl");
        if let Some(type_params) = &type_params {
            format_to!(buf, "{}", type_params.syntax());
        }
        buf.push_str(" ");
        buf.push_str(name.text().as_str());
        if let Some(type_params) = type_params {
            let lifetime_params = type_params
                .lifetime_params()
                .filter_map(|it| it.lifetime())
                .map(|it| it.text().clone());
            let type_params =
                type_params.type_params().filter_map(|it| it.name()).map(|it| it.text().clone());

            let generic_params = lifetime_params.chain(type_params).sep_by(", ");
            format_to!(buf, "<{}>", generic_params)
        }
        buf.push_str(" {\n");
        edit.set_cursor(start_offset + TextUnit::of_str(&buf));
        buf.push_str("\n}");
        edit.insert(start_offset, buf);
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::{check_assist, check_assist_target};

    #[test]
    fn test_add_impl() {
        check_assist(add_impl, "struct Foo {<|>}\n", "struct Foo {}\n\nimpl Foo {\n<|>\n}\n");
        check_assist(
            add_impl,
            "struct Foo<T: Clone> {<|>}",
            "struct Foo<T: Clone> {}\n\nimpl<T: Clone> Foo<T> {\n<|>\n}",
        );
        check_assist(
            add_impl,
            "struct Foo<'a, T: Foo<'a>> {<|>}",
            "struct Foo<'a, T: Foo<'a>> {}\n\nimpl<'a, T: Foo<'a>> Foo<'a, T> {\n<|>\n}",
        );
    }

    #[test]
    fn add_impl_target() {
        check_assist_target(
            add_impl,
            "
struct SomeThingIrrelevant;
/// Has a lifetime parameter
struct Foo<'a, T: Foo<'a>> {<|>}
struct EvenMoreIrrelevant;
",
            "/// Has a lifetime parameter
struct Foo<'a, T: Foo<'a>> {}",
        );
    }
}
