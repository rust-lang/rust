use hir::{Adt, ModuleDef};
use ide_db::defs::Definition;
use syntax::{
    ast::{self, AstNode, GenericParamsOwner, VisibilityOwner},
    match_ast,
};

use crate::{assist_context::AssistBuilder, AssistContext, AssistId, AssistKind, Assists};

// Assist: convert_tuple_struct_to_named_struct
//
// Converts tuple struct to struct with named fields.
//
// ```
// struct Inner;
// struct A$0(Inner);
// ```
// ->
// ```
// struct Inner;
// struct A { field1: Inner }
// ```
pub(crate) fn convert_tuple_struct_to_named_struct(
    acc: &mut Assists,
    ctx: &AssistContext,
) -> Option<()> {
    let strukt = ctx.find_node_at_offset::<ast::Struct>()?;
    let tuple_fields = match strukt.field_list()? {
        ast::FieldList::TupleFieldList(it) => it,
        ast::FieldList::RecordFieldList(_) => return None,
    };

    let target = strukt.syntax().text_range();
    acc.add(
        AssistId("convert_tuple_struct_to_named_struct", AssistKind::RefactorRewrite),
        "Convert to named struct",
        target,
        |edit| {
            let names = generate_names(tuple_fields.fields());
            edit_field_references(ctx, edit, tuple_fields.fields(), &names);
            edit_struct_references(ctx, edit, &strukt, &names);
            edit_struct_def(ctx, edit, &strukt, tuple_fields, names);
        },
    )
}

fn edit_struct_def(
    ctx: &AssistContext,
    edit: &mut AssistBuilder,
    strukt: &ast::Struct,
    tuple_fields: ast::TupleFieldList,
    names: Vec<ast::Name>,
) {
    let record_fields = tuple_fields
        .fields()
        .zip(names)
        .map(|(f, name)| ast::make::record_field(f.visibility(), name, f.ty().unwrap()));
    let record_fields = ast::make::record_field_list(record_fields);
    let tuple_fields_text_range = tuple_fields.syntax().text_range();

    edit.edit_file(ctx.frange.file_id);

    if let Some(w) = strukt.where_clause() {
        edit.delete(w.syntax().text_range());
        edit.insert(tuple_fields_text_range.start(), ast::make::tokens::single_newline().text());
        edit.insert(tuple_fields_text_range.start(), w.syntax().text());
        edit.insert(tuple_fields_text_range.start(), ",");
        edit.insert(tuple_fields_text_range.start(), ast::make::tokens::single_newline().text());
    } else {
        edit.insert(tuple_fields_text_range.start(), ast::make::tokens::single_space().text());
    }

    edit.replace(tuple_fields_text_range, record_fields.to_string());
    strukt.semicolon_token().map(|t| edit.delete(t.text_range()));
}

fn edit_struct_references(
    ctx: &AssistContext,
    edit: &mut AssistBuilder,
    strukt: &ast::Struct,
    names: &[ast::Name],
) {
    let strukt_def = ctx.sema.to_def(strukt).unwrap();
    let usages = Definition::ModuleDef(ModuleDef::Adt(Adt::Struct(strukt_def)))
        .usages(&ctx.sema)
        .include_self_kw_refs(true)
        .all();

    for (file_id, refs) in usages {
        edit.edit_file(file_id);
        for r in refs {
            for node in r.name.syntax().ancestors() {
                match_ast! {
                    match node {
                        ast::TupleStructPat(tuple_struct_pat) => {
                            edit.replace(
                                tuple_struct_pat.syntax().text_range(),
                                ast::make::record_pat_with_fields(
                                    tuple_struct_pat.path().unwrap(),
                                    ast::make::record_pat_field_list(tuple_struct_pat.fields().zip(names).map(
                                        |(pat, name)| {
                                            ast::make::record_pat_field(
                                                ast::make::name_ref(&name.to_string()),
                                                pat,
                                            )
                                        },
                                    )),
                                )
                                .to_string(),
                            );
                        },
                        // for tuple struct creations like: Foo(42)
                        ast::CallExpr(call_expr) => {
                            let path = call_expr.syntax().descendants().find_map(ast::PathExpr::cast).unwrap();
                            let arg_list =
                                call_expr.syntax().descendants().find_map(ast::ArgList::cast).unwrap();

                            edit.replace(
                                call_expr.syntax().text_range(),
                                ast::make::record_expr(
                                    path.path().unwrap(),
                                    ast::make::record_expr_field_list(arg_list.args().zip(names).map(
                                        |(expr, name)| {
                                            ast::make::record_expr_field(
                                                ast::make::name_ref(&name.to_string()),
                                                Some(expr),
                                            )
                                        },
                                    )),
                                )
                                .to_string(),
                            );
                        },
                        _ => ()
                    }
                }
            }
        }
    }
}

fn edit_field_references(
    ctx: &AssistContext,
    edit: &mut AssistBuilder,
    fields: impl Iterator<Item = ast::TupleField>,
    names: &[ast::Name],
) {
    for (field, name) in fields.zip(names) {
        let field = match ctx.sema.to_def(&field) {
            Some(it) => it,
            None => continue,
        };
        let def = Definition::Field(field);
        let usages = def.usages(&ctx.sema).all();
        for (file_id, refs) in usages {
            edit.edit_file(file_id);
            for r in refs {
                if let Some(name_ref) = r.name.as_name_ref() {
                    edit.replace(name_ref.syntax().text_range(), name.text());
                }
            }
        }
    }
}

fn generate_names(fields: impl Iterator<Item = ast::TupleField>) -> Vec<ast::Name> {
    fields.enumerate().map(|(i, _)| ast::make::name(&format!("field{}", i + 1))).collect()
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn not_applicable_other_than_tuple_struct() {
        check_assist_not_applicable(
            convert_tuple_struct_to_named_struct,
            r#"struct Foo$0 { bar: u32 };"#,
        );
        check_assist_not_applicable(convert_tuple_struct_to_named_struct, r#"struct Foo$0;"#);
    }

    #[test]
    fn convert_simple_struct() {
        check_assist(
            convert_tuple_struct_to_named_struct,
            r#"
struct Inner;
struct A$0(Inner);

impl A {
    fn new() -> A {
        A(Inner)
    }

    fn into_inner(self) -> Inner {
        self.0
    }
}"#,
            r#"
struct Inner;
struct A { field1: Inner }

impl A {
    fn new() -> A {
        A { field1: Inner }
    }

    fn into_inner(self) -> Inner {
        self.field1
    }
}"#,
        );
    }

    #[test]
    fn convert_struct_referenced_via_self_kw() {
        check_assist(
            convert_tuple_struct_to_named_struct,
            r#"
struct Inner;
struct A$0(Inner);

impl A {
    fn new() -> Self {
        Self(Inner)
    }

    fn into_inner(self) -> Inner {
        self.0
    }
}"#,
            r#"
struct Inner;
struct A { field1: Inner }

impl A {
    fn new() -> Self {
        Self { field1: Inner }
    }

    fn into_inner(self) -> Inner {
        self.field1
    }
}"#,
        );
    }

    #[test]
    fn convert_destructured_struct() {
        check_assist(
            convert_tuple_struct_to_named_struct,
            r#"
struct Inner;
struct A$0(Inner);

impl A {
    fn into_inner(self) -> Inner {
        let A(first) = self;
        first
    }

    fn into_inner_via_self(self) -> Inner {
        let Self(first) = self;
        first
    }
}"#,
            r#"
struct Inner;
struct A { field1: Inner }

impl A {
    fn into_inner(self) -> Inner {
        let A { field1: first } = self;
        first
    }

    fn into_inner_via_self(self) -> Inner {
        let Self { field1: first } = self;
        first
    }
}"#,
        );
    }

    #[test]
    fn convert_struct_with_visibility() {
        check_assist(
            convert_tuple_struct_to_named_struct,
            r#"
struct A$0(pub u32, pub(crate) u64);

impl A {
    fn new() -> A {
        A(42, 42)
    }

    fn into_first(self) -> u32 {
        self.0
    }

    fn into_second(self) -> u64 {
        self.1
    }
}"#,
            r#"
struct A { pub field1: u32, pub(crate) field2: u64 }

impl A {
    fn new() -> A {
        A { field1: 42, field2: 42 }
    }

    fn into_first(self) -> u32 {
        self.field1
    }

    fn into_second(self) -> u64 {
        self.field2
    }
}"#,
        );
    }

    #[test]
    fn convert_struct_with_where_clause() {
        check_assist(
            convert_tuple_struct_to_named_struct,
            r#"
struct Wrap$0<T>(T)
where
    T: Display;
"#,
            r#"
struct Wrap<T>
where
    T: Display,
{ field1: T }

"#,
        );
    }
}
