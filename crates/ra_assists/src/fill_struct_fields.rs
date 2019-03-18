use std::fmt::Write;

use hir::{AdtDef, Ty, source_binder};
use hir::db::HirDatabase;

use ra_syntax::ast::{self, AstNode};

use crate::{AssistCtx, Assist, AssistId};

pub(crate) fn fill_struct_fields(mut ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let struct_lit = ctx.node_at_offset::<ast::StructLit>()?;
    let named_field_list = struct_lit.named_field_list()?;

    // If we already have existing struct fields, don't provide the assist.
    if named_field_list.fields().count() > 0 {
        return None;
    }

    let function =
        source_binder::function_from_child_node(ctx.db, ctx.frange.file_id, struct_lit.syntax())?;

    let infer_result = function.infer(ctx.db);
    let source_map = function.body_source_map(ctx.db);
    let node_expr = source_map.node_expr(struct_lit.into())?;
    let struct_lit_ty = infer_result[node_expr].clone();
    let struct_def = match struct_lit_ty {
        Ty::Adt { def_id: AdtDef::Struct(s), .. } => s,
        _ => return None,
    };

    let db = ctx.db;
    ctx.add_action(AssistId("fill_struct_fields"), "fill struct fields", |edit| {
        let mut buf = String::from("{\n");
        let struct_fields = struct_def.fields(db);
        for field in struct_fields {
            let field_name = field.name(db).to_string();
            write!(&mut buf, "    {}: (),\n", field_name).unwrap();
        }
        buf.push_str("}");

        edit.target(struct_lit.syntax().range());
        edit.set_cursor(struct_lit.syntax().range().start());
        edit.replace_node_and_indent(named_field_list.syntax(), buf);
    });

    ctx.build()
}

#[cfg(test)]
mod tests {
    use crate::helpers::{check_assist, check_assist_target};

    use super::fill_struct_fields;

    #[test]
    fn fill_struct_fields_empty_body() {
        check_assist(
            fill_struct_fields,
            r#"
            struct S<'a, D> {
                a: u32,
                b: String,
                c: (i32, i32),
                d: D,
                r: &'a str,
            }

            fn main() {
                let s = S<|> {}
            }
            "#,
            r#"
            struct S<'a, D> {
                a: u32,
                b: String,
                c: (i32, i32),
                d: D,
                r: &'a str,
            }

            fn main() {
                let s = <|>S {
                    a: (),
                    b: (),
                    c: (),
                    d: (),
                    r: (),
                }
            }
            "#,
        );
    }

    #[test]
    fn fill_struct_fields_target() {
        check_assist_target(
            fill_struct_fields,
            r#"
            struct S<'a, D> {
                a: u32,
                b: String,
                c: (i32, i32),
                d: D,
                r: &'a str,
            }

            fn main() {
                let s = S<|> {}
            }
            "#,
            "S {}",
        );
    }

    #[test]
    fn fill_struct_fields_preserve_self() {
        check_assist(
            fill_struct_fields,
            r#"
            struct Foo {
                foo: u8,
                bar: String,
                baz: i128,
            }

            impl Foo {
                pub fn new() -> Self {
                    Self <|>{}
                }
            }
            "#,
            r#"
            struct Foo {
                foo: u8,
                bar: String,
                baz: i128,
            }

            impl Foo {
                pub fn new() -> Self {
                    <|>Self {
                        foo: (),
                        bar: (),
                        baz: (),
                    }
                }
            }
            "#,
        );
    }
}
