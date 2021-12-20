use ide_db::helpers::FamousDefs;
use stdx::{format_to, to_lower_snake_case};
use syntax::ast::{self, AstNode, HasName, HasVisibility};

use crate::{
    utils::{convert_reference_type, find_impl_block_end, find_struct_impl, generate_impl_text},
    AssistContext, AssistId, AssistKind, Assists, GroupLabel,
};

// Assist: generate_getter
//
// Generate a getter method.
//
// ```
// # //- minicore: as_ref
// # pub struct String;
// # impl AsRef<str> for String {
// #     fn as_ref(&self) -> &str {
// #         ""
// #     }
// # }
// #
// struct Person {
//     nam$0e: String,
// }
// ```
// ->
// ```
// # pub struct String;
// # impl AsRef<str> for String {
// #     fn as_ref(&self) -> &str {
// #         ""
// #     }
// # }
// #
// struct Person {
//     name: String,
// }
//
// impl Person {
//     /// Get a reference to the person's name.
//     fn $0name(&self) -> &str {
//         self.name.as_ref()
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
            let (ty, body, description) = if mutable {
                (
                    format!("&mut {}", field_ty),
                    format!("&mut self.{}", field_name),
                    "a mutable reference to ",
                )
            } else {
                let famous_defs = &FamousDefs(&ctx.sema, ctx.sema.scope(field_ty.syntax()).krate());
                ctx.sema
                    .resolve_type(&field_ty)
                    .and_then(|ty| convert_reference_type(ty, ctx.db(), famous_defs))
                    .map(|conversion| {
                        cov_mark::hit!(convert_reference_type);
                        (
                            conversion.convert_type(ctx.db()),
                            conversion.getter(field_name.to_string()),
                            if conversion.is_copy() { "" } else { "a reference to " },
                        )
                    })
                    .unwrap_or_else(|| {
                        (
                            format!("&{}", field_ty),
                            format!("&self.{}", field_name),
                            "a reference to ",
                        )
                    })
            };

            format_to!(
                buf,
                "    /// Get {}the {}'s {}.
    {}fn {}(&{}self) -> {} {{
        {}
    }}",
                description,
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
    fn test_not_a_special_case() {
        cov_mark::check_count!(convert_reference_type, 0);
        // Fake string which doesn't implement AsRef<str>
        check_assist(
            generate_getter,
            r#"
pub struct String;

struct S { foo: $0String }
"#,
            r#"
pub struct String;

struct S { foo: String }

impl S {
    /// Get a reference to the s's foo.
    fn $0foo(&self) -> &String {
        &self.foo
    }
}
"#,
        );
    }

    #[test]
    fn test_convert_reference_type() {
        cov_mark::check_count!(convert_reference_type, 6);

        // Copy
        check_assist(
            generate_getter,
            r#"
//- minicore: copy
struct S { foo: $0bool }
"#,
            r#"
struct S { foo: bool }

impl S {
    /// Get the s's foo.
    fn $0foo(&self) -> bool {
        self.foo
    }
}
"#,
        );

        // AsRef<str>
        check_assist(
            generate_getter,
            r#"
//- minicore: as_ref
pub struct String;
impl AsRef<str> for String {
    fn as_ref(&self) -> &str {
        ""
    }
}

struct S { foo: $0String }
"#,
            r#"
pub struct String;
impl AsRef<str> for String {
    fn as_ref(&self) -> &str {
        ""
    }
}

struct S { foo: String }

impl S {
    /// Get a reference to the s's foo.
    fn $0foo(&self) -> &str {
        self.foo.as_ref()
    }
}
"#,
        );

        // AsRef<T>
        check_assist(
            generate_getter,
            r#"
//- minicore: as_ref
struct Sweets;

pub struct Box<T>(T);
impl<T> AsRef<T> for Box<T> {
    fn as_ref(&self) -> &T {
        &self.0
    }
}

struct S { foo: $0Box<Sweets> }
"#,
            r#"
struct Sweets;

pub struct Box<T>(T);
impl<T> AsRef<T> for Box<T> {
    fn as_ref(&self) -> &T {
        &self.0
    }
}

struct S { foo: Box<Sweets> }

impl S {
    /// Get a reference to the s's foo.
    fn $0foo(&self) -> &Sweets {
        self.foo.as_ref()
    }
}
"#,
        );

        // AsRef<[T]>
        check_assist(
            generate_getter,
            r#"
//- minicore: as_ref
pub struct Vec<T>;
impl<T> AsRef<[T]> for Vec<T> {
    fn as_ref(&self) -> &[T] {
        &[]
    }
}

struct S { foo: $0Vec<()> }
"#,
            r#"
pub struct Vec<T>;
impl<T> AsRef<[T]> for Vec<T> {
    fn as_ref(&self) -> &[T] {
        &[]
    }
}

struct S { foo: Vec<()> }

impl S {
    /// Get a reference to the s's foo.
    fn $0foo(&self) -> &[()] {
        self.foo.as_ref()
    }
}
"#,
        );

        // Option
        check_assist(
            generate_getter,
            r#"
//- minicore: option
struct Failure;

struct S { foo: $0Option<Failure> }
"#,
            r#"
struct Failure;

struct S { foo: Option<Failure> }

impl S {
    /// Get a reference to the s's foo.
    fn $0foo(&self) -> Option<&Failure> {
        self.foo.as_ref()
    }
}
"#,
        );

        // Result
        check_assist(
            generate_getter,
            r#"
//- minicore: result
struct Context {
    dat$0a: Result<bool, i32>,
}
"#,
            r#"
struct Context {
    data: Result<bool, i32>,
}

impl Context {
    /// Get a reference to the context's data.
    fn $0data(&self) -> Result<&bool, &i32> {
        self.data.as_ref()
    }
}
"#,
        );
    }
}
