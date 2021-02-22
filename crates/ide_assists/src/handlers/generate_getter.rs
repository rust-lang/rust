use stdx::{format_to, to_lower_snake_case};
use syntax::ast::{self, AstNode, NameOwner, VisibilityOwner};

use crate::{
    utils::{find_impl_block_end, find_struct_impl, generate_impl_text},
    AssistContext, AssistId, AssistKind, Assists, GroupLabel,
};

// Assist: generate_getter
//
// Generate a getter method.
//
// ```
// struct Person {
//     nam$0e: String,
// }
// ```
// ->
// ```
// struct Person {
//     name: String,
// }
//
// impl Person {
//     /// Get a reference to the person's name.
//     fn name(&self) -> &String {
//         &self.name
//     }
// }
// ```
pub(crate) fn generate_getter(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    let strukt = ctx.find_node_at_offset::<ast::Struct>()?;
    let field = ctx.find_node_at_offset::<ast::RecordField>()?;

    let strukt_name = strukt.name()?;
    let field_name = field.name()?;
    let field_ty = field.ty()?;

    // Return early if we've found an existing fn
    let fn_name = to_lower_snake_case(&field_name.to_string());
    let impl_def = find_struct_impl(&ctx, &ast::Adt::Struct(strukt.clone()), fn_name.as_str())?;

    let target = field.syntax().text_range();
    acc.add_group(
        &GroupLabel("Generate getter/setter".to_owned()),
        AssistId("generate_getter", AssistKind::Generate),
        "Generate a getter method",
        target,
        |builder| {
            let mut buf = String::with_capacity(512);

            let fn_name_spaced = fn_name.replace('_', " ");
            let strukt_name_spaced =
                to_lower_snake_case(&strukt_name.to_string()).replace('_', " ");

            if impl_def.is_some() {
                buf.push('\n');
            }

            let vis = strukt.visibility().map_or(String::new(), |v| format!("{} ", v));
            format_to!(
                buf,
                "    /// Get a reference to the {}'s {}.
    {}fn {}(&self) -> &{} {{
        &self.{}
    }}",
                strukt_name_spaced,
                fn_name_spaced,
                vis,
                fn_name,
                field_ty,
                fn_name,
            );

            let start_offset = impl_def
                .and_then(|impl_def| find_impl_block_end(impl_def, &mut buf))
                .unwrap_or_else(|| {
                    buf = generate_impl_text(&ast::Adt::Struct(strukt.clone()), &buf);
                    strukt.syntax().text_range().end()
                });

            builder.insert(start_offset, buf);
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    fn check_not_applicable(ra_fixture: &str) {
        check_assist_not_applicable(generate_getter, ra_fixture)
    }

    #[test]
    fn test_generate_getter_from_field() {
        check_assist(
            generate_getter,
            r#"
struct Context<T: Clone> {
    dat$0a: T,
}"#,
            r#"
struct Context<T: Clone> {
    data: T,
}

impl<T: Clone> Context<T> {
    /// Get a reference to the context's data.
    fn data(&self) -> &T {
        &self.data
    }
}"#,
        );
    }

    #[test]
    fn test_generate_getter_already_implemented() {
        check_not_applicable(
            r#"
struct Context<T: Clone> {
    dat$0a: T,
}

impl<T: Clone> Context<T> {
    fn data(&self) -> &T {
        &self.data
    }
}"#,
        );
    }

    #[test]
    fn test_generate_getter_from_field_with_visibility_marker() {
        check_assist(
            generate_getter,
            r#"
pub(crate) struct Context<T: Clone> {
    dat$0a: T,
}"#,
            r#"
pub(crate) struct Context<T: Clone> {
    data: T,
}

impl<T: Clone> Context<T> {
    /// Get a reference to the context's data.
    pub(crate) fn data(&self) -> &T {
        &self.data
    }
}"#,
        );
    }

    #[test]
    fn test_multiple_generate_getter() {
        check_assist(
            generate_getter,
            r#"
struct Context<T: Clone> {
    data: T,
    cou$0nt: usize,
}

impl<T: Clone> Context<T> {
    /// Get a reference to the context's data.
    fn data(&self) -> &T {
        &self.data
    }
}"#,
            r#"
struct Context<T: Clone> {
    data: T,
    count: usize,
}

impl<T: Clone> Context<T> {
    /// Get a reference to the context's data.
    fn data(&self) -> &T {
        &self.data
    }

    /// Get a reference to the context's count.
    fn count(&self) -> &usize {
        &self.count
    }
}"#,
        );
    }
}
