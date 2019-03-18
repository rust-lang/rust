use std::fmt::Write;

use hir::{AdtDef, Ty, source_binder};
use hir::db::HirDatabase;

use ra_syntax::ast::{self, AstNode, Expr};

use crate::{AssistCtx, Assist, AssistId};

pub(crate) fn fill_struct_fields(mut ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let struct_lit = ctx.node_at_offset::<ast::StructLit>()?;

    // If we already have existing struct fields, don't provide the assist.
    match struct_lit.named_field_list() {
        Some(named_field_list) if named_field_list.fields().count() > 0 => {
            return None;
        }
        _ => {}
    }

    let expr: &Expr = struct_lit.into();
    let function =
        source_binder::function_from_child_node(ctx.db, ctx.frange.file_id, struct_lit.syntax())?;

    let infer_result = function.infer(ctx.db);
    let source_map = function.body_source_map(ctx.db);
    let node_expr = source_map.node_expr(expr)?;
    let struct_lit_ty = infer_result[node_expr].clone();
    let struct_def = match struct_lit_ty {
        Ty::Adt { def_id: AdtDef::Struct(s), .. } => s,
        _ => return None,
    };

    let struct_name = struct_def.name(ctx.db)?;
    let db = ctx.db;

    ctx.add_action(AssistId("fill_struct_fields"), "fill struct fields", |edit| {
        let mut buf = format!("{} {{\n", struct_name);
        let struct_fields = struct_def.fields(db);
        for field in struct_fields {
            let field_name = field.name(db).to_string();
            write!(&mut buf, "    {}: (),\n", field_name).unwrap();
        }
        buf.push_str("}");

        edit.target(struct_lit.syntax().range());
        edit.set_cursor(expr.syntax().range().start());
        edit.replace_node_and_indent(struct_lit.syntax(), buf);
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
}
