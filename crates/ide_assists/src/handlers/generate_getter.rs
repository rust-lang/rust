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
    generate_getter_impl(acc, ctx, false)
}

// Assist: generate_getter_mut
//
// Generate a mut getter method.
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
//     /// Get a mutable reference to the person's name.
//     fn name_mut(&mut self) -> &mut String {
//         &mut self.name
//     }
// }
// ```
pub(crate) fn generate_getter_mut(acc: &mut Assists, ctx: &AssistContext) -> Option<()> {
    generate_getter_impl(acc, ctx, true)
}

pub(crate) fn generate_getter_impl(
    acc: &mut Assists,
    ctx: &AssistContext,
    mutable: bool,
) -> Option<()> {
    let strukt = ctx.find_node_at_offset::<ast::Struct>()?;
    let field = ctx.find_node_at_offset::<ast::RecordField>()?;

    let strukt_name = strukt.name()?;
    let field_name = field.name()?;
    let field_ty = field.ty()?;

    // Return early if we've found an existing fn
    let mut fn_name = to_lower_snake_case(&field_name.to_string());
    if mutable {
        format_to!(fn_name, "_mut");
    }
    let impl_def = find_struct_impl(&ctx, &ast::Adt::Struct(strukt.clone()), fn_name.as_str())?;

    let (id, label) = if mutable {
        ("generate_getter_mut", "Generate a mut getter method")
    } else {
        ("generate_getter", "Generate a getter method")
    };
    let target = field.syntax().text_range();
    acc.add_group(
        &GroupLabel("Generate getter/setter".to_owned()),
        AssistId(id, AssistKind::Generate),
        label,
        target,
        |builder| {
            let mut buf = String::with_capacity(512);

            if impl_def.is_some() {
                buf.push('\n');
            }

            let vis = strukt.visibility().map_or(String::new(), |v| format!("{} ", v));
            format_to!(
                buf,
                "    /// Get a {}reference to the {}'s {}.
    {}fn {}(&{mut_}self) -> &{mut_}{} {{
        &{mut_}self.{}
    }}",
                mutable.then(|| "mutable ").unwrap_or_default(),
                to_lower_snake_case(&strukt_name.to_string()).replace('_', " "),
                fn_name.trim_end_matches("_mut").replace('_', " "),
                vis,
                fn_name,
                field_ty,
                field_name,
                mut_ = mutable.then(|| "mut ").unwrap_or_default(),
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

    #[test]
    fn test_generate_getter_from_field() {
        check_assist(
            generate_getter,
            r#"
struct Context {
    dat$0a: Data,
}
"#,
            r#"
struct Context {
    data: Data,
}

impl Context {
    /// Get a reference to the context's data.
    fn data(&self) -> &Data {
        &self.data
    }
}
"#,
        );

        check_assist(
            generate_getter_mut,
            r#"
struct Context {
    dat$0a: Data,
}
"#,
            r#"
struct Context {
    data: Data,
}

impl Context {
    /// Get a mutable reference to the context's data.
    fn data_mut(&mut self) -> &mut Data {
        &mut self.data
    }
}
"#,
        );
    }

    #[test]
    fn test_generate_getter_already_implemented() {
        check_assist_not_applicable(
            generate_getter,
            r#"
struct Context {
    dat$0a: Data,
}

impl Context {
    fn data(&self) -> &Data {
        &self.data
    }
}
"#,
        );

        check_assist_not_applicable(
            generate_getter_mut,
            r#"
struct Context {
    dat$0a: Data,
}

impl Context {
    fn data_mut(&mut self) -> &mut Data {
        &mut self.data
    }
}
"#,
        );
    }

    #[test]
    fn test_generate_getter_from_field_with_visibility_marker() {
        check_assist(
            generate_getter,
            r#"
pub(crate) struct Context {
    dat$0a: Data,
}
"#,
            r#"
pub(crate) struct Context {
    data: Data,
}

impl Context {
    /// Get a reference to the context's data.
    pub(crate) fn data(&self) -> &Data {
        &self.data
    }
}
"#,
        );
    }

    #[test]
    fn test_multiple_generate_getter() {
        check_assist(
            generate_getter,
            r#"
struct Context {
    data: Data,
    cou$0nt: usize,
}

impl Context {
    /// Get a reference to the context's data.
    fn data(&self) -> &Data {
        &self.data
    }
}
"#,
            r#"
struct Context {
    data: Data,
    count: usize,
}

impl Context {
    /// Get a reference to the context's data.
    fn data(&self) -> &Data {
        &self.data
    }

    /// Get a reference to the context's count.
    fn count(&self) -> &usize {
        &self.count
    }
}
"#,
        );
    }
}
