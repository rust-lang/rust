use std::fmt::Write;

use hir::{AdtDef, db::HirDatabase};

use ra_syntax::ast::{self, AstNode};

use crate::{AssistCtx, Assist, AssistId};

pub(crate) fn fill_struct_fields(mut ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    let struct_lit = ctx.node_at_offset::<ast::StructLit>()?;
    let mut fsf = FillStructFields {
        ctx: &mut ctx,
        named_field_list: struct_lit.named_field_list()?,
        struct_fields: vec![],
        struct_lit,
    };
    fsf.evaluate_struct_def_fields()?;
    if fsf.struct_lit_and_def_have_the_same_number_of_fields() {
        return None;
    }
    fsf.remove_already_included_fields()?;
    fsf.add_action()?;
    ctx.build()
}

struct FillStructFields<'a, 'b: 'a, DB> {
    ctx: &'a mut AssistCtx<'b, DB>,
    named_field_list: &'a ast::NamedFieldList,
    struct_fields: Vec<(String, String)>,
    struct_lit: &'a ast::StructLit,
}

impl<DB> FillStructFields<'_, '_, DB>
where
    DB: HirDatabase,
{
    fn add_action(&mut self) -> Option<()> {
        let named_field_list = self.named_field_list;
        let struct_fields_string = self.struct_fields_string()?;
        let struct_lit = self.struct_lit;
        self.ctx.add_action(AssistId("fill_struct_fields"), "fill struct fields", |edit| {
            edit.target(struct_lit.syntax().range());
            edit.set_cursor(struct_lit.syntax().range().start());
            edit.replace_node_and_indent(named_field_list.syntax(), struct_fields_string);
        });
        Some(())
    }

    fn struct_lit_and_def_have_the_same_number_of_fields(&self) -> bool {
        self.named_field_list.fields().count() == self.struct_fields.len()
    }

    fn evaluate_struct_def_fields(&mut self) -> Option<()> {
        let analyzer = hir::SourceAnalyzer::new(
            self.ctx.db,
            self.ctx.frange.file_id,
            self.struct_lit.syntax(),
            None,
        );
        let struct_lit_ty = analyzer.type_of(self.ctx.db, self.struct_lit.into())?;
        let struct_def = match struct_lit_ty.as_adt() {
            Some((AdtDef::Struct(s), _)) => s,
            _ => return None,
        };
        self.struct_fields = struct_def
            .fields(self.ctx.db)
            .into_iter()
            .map(|f| (f.name(self.ctx.db).to_string(), "()".into()))
            .collect();
        Some(())
    }

    fn remove_already_included_fields(&mut self) -> Option<()> {
        for ast_field in self.named_field_list.fields() {
            let expr = ast_field.expr()?.syntax().text().to_string();
            let name_from_ast = ast_field.name_ref()?.text().to_string();
            if let Some(idx) = self.struct_fields.iter().position(|(n, _)| n == &name_from_ast) {
                self.struct_fields[idx] = (name_from_ast, expr);
            }
        }
        Some(())
    }

    fn struct_fields_string(&mut self) -> Option<String> {
        let mut buf = String::from("{\n");
        for (name, expr) in &self.struct_fields {
            write!(&mut buf, "    {}: {},\n", name, expr).unwrap();
        }
        buf.push_str("}");
        Some(buf)
    }
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
                    a: (),
                    b: (),
                    c: (1, 2),
                    d: (),
                    e: "foo",
                }
            }
            "#,
        );
    }
}
