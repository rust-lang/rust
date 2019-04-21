use hir::{AdtDef, db::HirDatabase};

use ra_syntax::ast::{self, AstNode};

use crate::{AssistCtx, Assist, AssistId, ast_editor::{AstEditor, AstBuilder}};

pub(crate) fn fill_struct_fields(mut ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let struct_lit = ctx.node_at_offset::<ast::StructLit>()?;
    let named_field_list = struct_lit.named_field_list()?;

    // Collect all fields from struct definition
    let mut fields = {
        let analyzer =
            hir::SourceAnalyzer::new(ctx.db, ctx.frange.file_id, struct_lit.syntax(), None);
        let struct_lit_ty = analyzer.type_of(ctx.db, struct_lit.into())?;
        let struct_def = match struct_lit_ty.as_adt() {
            Some((AdtDef::Struct(s), _)) => s,
            _ => return None,
        };
        struct_def.fields(ctx.db)
    };

    // Filter out existing fields
    for ast_field in named_field_list.fields() {
        let name_from_ast = ast_field.name_ref()?.text().to_string();
        fields.retain(|field| field.name(ctx.db).to_string() != name_from_ast);
    }
    if fields.is_empty() {
        return None;
    }

    let db = ctx.db;
    ctx.add_action(AssistId("fill_struct_fields"), "fill struct fields", |edit| {
        let mut ast_editor = AstEditor::new(named_field_list);
        if named_field_list.fields().count() == 0 && fields.len() > 2 {
            ast_editor.make_multiline();
        };

        for field in fields {
            let field =
                AstBuilder::<ast::NamedField>::from_text(&format!("{}: ()", field.name(db)));
            ast_editor.append_field(&field);
        }

        edit.target(struct_lit.syntax().range());
        edit.set_cursor(struct_lit.syntax().range().start());

        ast_editor.into_text_edit(edit.text_edit_builder());
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
                e: &'a str,
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
                e: &'a str,
            }

            fn main() {
                let s = <|>S {
                    a: (),
                    b: (),
                    c: (),
                    d: (),
                    e: (),
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
                e: &'a str,
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

    #[test]
    fn fill_struct_fields_partial() {
        check_assist(
            fill_struct_fields,
            r#"
            struct S<'a, D> {
                a: u32,
                b: String,
                c: (i32, i32),
                d: D,
                e: &'a str,
            }

            fn main() {
                let s = S {
                    c: (1, 2),
                    e: "foo",<|>
                }
            }
            "#,
            r#"
            struct S<'a, D> {
                a: u32,
                b: String,
                c: (i32, i32),
                d: D,
                e: &'a str,
            }

            fn main() {
                let s = <|>S {
                    c: (1, 2),
                    e: "foo",
                    a: (),
                    b: (),
                    d: (),
                }
            }
            "#,
        );
    }
}
