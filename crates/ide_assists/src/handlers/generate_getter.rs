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
//     fn $0name(&self) -> &str {
//         self.name.as_str()
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
//     fn $0name_mut(&mut self) -> &mut String {
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
    let impl_def = find_struct_impl(ctx, &ast::Adt::Struct(strukt.clone()), fn_name.as_str())?;

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
            let (ty, body) = if mutable {
                (format!("&mut {}", field_ty), format!("&mut self.{}", field_name))
            } else {
                useless_type_special_case(&field_name.to_string(), &field_ty)
                    .unwrap_or_else(|| (format!("&{}", field_ty), format!("&self.{}", field_name)))
            };

            format_to!(
                buf,
                "    /// Get a {}reference to the {}'s {}.
    {}fn {}(&{}self) -> {} {{
        {}
    }}",
                mutable.then(|| "mutable ").unwrap_or_default(),
                to_lower_snake_case(&strukt_name.to_string()).replace('_', " "),
                fn_name.trim_end_matches("_mut").replace('_', " "),
                vis,
                fn_name,
                mutable.then(|| "mut ").unwrap_or_default(),
                ty,
                body,
            );

            let start_offset = impl_def
                .and_then(|impl_def| find_impl_block_end(impl_def, &mut buf))
                .unwrap_or_else(|| {
                    buf = generate_impl_text(&ast::Adt::Struct(strukt.clone()), &buf);
                    strukt.syntax().text_range().end()
                });

            match ctx.config.snippet_cap {
                Some(cap) => {
                    builder.insert_snippet(cap, start_offset, buf.replacen("fn ", "fn $0", 1))
                }
                None => builder.insert(start_offset, buf),
            }
        },
    )
}

fn useless_type_special_case(field_name: &str, field_ty: &ast::Type) -> Option<(String, String)> {
    if field_ty.to_string() == "String" {
        cov_mark::hit!(useless_type_special_case);
        return Some(("&str".to_string(), format!("self.{}.as_str()", field_name)));
    }
    if let Some(arg) = ty_ctor(field_ty, "Vec") {
        return Some((format!("&[{}]", arg), format!("self.{}.as_slice()", field_name)));
    }
    if let Some(arg) = ty_ctor(field_ty, "Box") {
        return Some((format!("&{}", arg), format!("self.{}.as_ref()", field_name)));
    }
    if let Some(arg) = ty_ctor(field_ty, "Option") {
        return Some((format!("Option<&{}>", arg), format!("self.{}.as_ref()", field_name)));
    }
    None
}

// FIXME: This should rely on semantic info.
fn ty_ctor(ty: &ast::Type, ctor: &str) -> Option<String> {
    let res = ty.to_string().strip_prefix(ctor)?.strip_prefix('<')?.strip_suffix('>')?.to_string();
    Some(res)
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
    fn $0data(&self) -> &Data {
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
    fn $0data_mut(&mut self) -> &mut Data {
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
    pub(crate) fn $0data(&self) -> &Data {
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
    fn $0count(&self) -> &usize {
        &self.count
    }
}
"#,
        );
    }

    #[test]
    fn test_special_cases() {
        cov_mark::check!(useless_type_special_case);
        check_assist(
            generate_getter,
            r#"
struct S { foo: $0String }
"#,
            r#"
struct S { foo: String }

impl S {
    /// Get a reference to the s's foo.
    fn $0foo(&self) -> &str {
        self.foo.as_str()
    }
}
"#,
        );
        check_assist(
            generate_getter,
            r#"
struct S { foo: $0Box<Sweets> }
"#,
            r#"
struct S { foo: Box<Sweets> }

impl S {
    /// Get a reference to the s's foo.
    fn $0foo(&self) -> &Sweets {
        self.foo.as_ref()
    }
}
"#,
        );
        check_assist(
            generate_getter,
            r#"
struct S { foo: $0Vec<()> }
"#,
            r#"
struct S { foo: Vec<()> }

impl S {
    /// Get a reference to the s's foo.
    fn $0foo(&self) -> &[()] {
        self.foo.as_slice()
    }
}
"#,
        );
        check_assist(
            generate_getter,
            r#"
struct S { foo: $0Option<Failure> }
"#,
            r#"
struct S { foo: Option<Failure> }

impl S {
    /// Get a reference to the s's foo.
    fn $0foo(&self) -> Option<&Failure> {
        self.foo.as_ref()
    }
}
"#,
        );
    }
}
