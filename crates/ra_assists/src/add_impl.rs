use join_to_string::join;
use hir::db::HirDatabase;
use ra_syntax::{
    ast::{self, AstNode, AstToken, NameOwner, TypeParamsOwner},
    TextUnit,
};

use crate::{AssistCtx, Assist};

pub(crate) fn add_impl(ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let nominal = ctx.node_at_offset::<ast::NominalDef>()?;
    let name = nominal.name()?;
    ctx.build("add impl", |edit| {
        let type_params = nominal.type_param_list();
        let start_offset = nominal.syntax().range().end();
        let mut buf = String::new();
        buf.push_str("\n\nimpl");
        if let Some(type_params) = type_params {
            type_params.syntax().text().push_to(&mut buf);
        }
        buf.push_str(" ");
        buf.push_str(name.text().as_str());
        if let Some(type_params) = type_params {
            let lifetime_params =
                type_params.lifetime_params().filter_map(|it| it.lifetime()).map(|it| it.text());
            let type_params =
                type_params.type_params().filter_map(|it| it.name()).map(|it| it.text());
            join(lifetime_params.chain(type_params)).surround_with("<", ">").to_buf(&mut buf);
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
    use crate::helpers::check_assist;

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

}
