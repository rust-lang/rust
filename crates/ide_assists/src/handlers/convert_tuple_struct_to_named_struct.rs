use ide_db::defs::{Definition, NameRefClass};
use syntax::{
    ast::{self, AstNode, GenericParamsOwner, VisibilityOwner},
    match_ast, SyntaxNode,
};

use crate::{assist_context::AssistBuilder, AssistContext, AssistId, AssistKind, Assists};

// Assist: convert_tuple_struct_to_named_struct
//
// Converts tuple struct to struct with named fields.
//
// ```
// struct Point$0(f32, f32);
//
// impl Point {
//     pub fn new(x: f32, y: f32) -> Self {
//         Point(x, y)
//     }
//
//     pub fn x(&self) -> f32 {
//         self.0
//     }
//
//     pub fn y(&self) -> f32 {
//         self.1
//     }
// }
// ```
// ->
// ```
// struct Point { field1: f32, field2: f32 }
//
// impl Point {
//     pub fn new(x: f32, y: f32) -> Self {
//         Point { field1: x, field2: y }
//     }
//
//     pub fn x(&self) -> f32 {
//         self.field1
//     }
//
//     pub fn y(&self) -> f32 {
//         self.field2
//     }
// }
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
    let strukt_def = ctx.sema.to_def(&strukt)?;

    let target = strukt.syntax().text_range();
    acc.add(
        AssistId("convert_tuple_struct_to_named_struct", AssistKind::RefactorRewrite),
        "Convert to named struct",
        target,
        |edit| {
            let names = generate_names(tuple_fields.fields());
            edit_field_references(ctx, edit, tuple_fields.fields(), &names);
            edit_struct_references(ctx, edit, strukt_def, &names);
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
        .filter_map(|(f, name)| Some(ast::make::record_field(f.visibility(), name, f.ty()?)));
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
    strukt: hir::Struct,
    names: &[ast::Name],
) {
    let strukt_def = Definition::ModuleDef(hir::ModuleDef::Adt(hir::Adt::Struct(strukt)));
    let usages = strukt_def.usages(&ctx.sema).include_self_kw_refs(true).all();

    let edit_node = |edit: &mut AssistBuilder, node: SyntaxNode| -> Option<()> {
        match_ast! {
            match node {
                ast::TupleStructPat(tuple_struct_pat) => {
                    edit.replace(
                        tuple_struct_pat.syntax().text_range(),
                        ast::make::record_pat_with_fields(
                            tuple_struct_pat.path()?,
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
                // for tuple struct creations like Foo(42)
                ast::CallExpr(call_expr) => {
                    let path = call_expr.syntax().descendants().find_map(ast::PathExpr::cast).and_then(|expr| expr.path())?;

                    // this also includes method calls like Foo::new(42), we should skip them
                    if let Some(name_ref) = path.segment().and_then(|s| s.name_ref()) {
                        match NameRefClass::classify(&ctx.sema, &name_ref) {
                            Some(NameRefClass::Definition(Definition::SelfType(_))) => {},
                            Some(NameRefClass::Definition(def)) if def == strukt_def => {},
                            _ => return None,
                        };
                    }

                    let arg_list = call_expr.syntax().descendants().find_map(ast::ArgList::cast)?;

                    edit.replace(
                        call_expr.syntax().text_range(),
                        ast::make::record_expr(
                            path,
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
                _ => return None,
            }
        }
        Some(())
    };

    for (file_id, refs) in usages {
        edit.edit_file(file_id);
        for r in refs {
            for node in r.name.syntax().ancestors() {
                if edit_node(edit, node).is_some() {
                    break;
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
    fn new(inner: Inner) -> A {
        A(inner)
    }

    fn new_with_default() -> A {
        A::new(Inner)
    }

    fn into_inner(self) -> Inner {
        self.0
    }
}"#,
            r#"
struct Inner;
struct A { field1: Inner }

impl A {
    fn new(inner: Inner) -> A {
        A { field1: inner }
    }

    fn new_with_default() -> A {
        A::new(Inner)
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
    fn new(inner: Inner) -> Self {
        Self(inner)
    }

    fn new_with_default() -> Self {
        Self::new(Inner)
    }

    fn into_inner(self) -> Inner {
        self.0
    }
}"#,
            r#"
struct Inner;
struct A { field1: Inner }

impl A {
    fn new(inner: Inner) -> Self {
        Self { field1: inner }
    }

    fn new_with_default() -> Self {
        Self::new(Inner)
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
    fn convert_struct_with_wrapped_references() {
        check_assist(
            convert_tuple_struct_to_named_struct,
            r#"
struct Inner$0(u32);
struct Outer(Inner);

impl Outer {
    fn new() -> Self {
        Self(Inner(42))
    }

    fn into_inner(self) -> u32 {
        (self.0).0
    }

    fn into_inner_destructed(self) -> u32 {
        let Outer(Inner(x)) = self;
        x
    }
}"#,
            r#"
struct Inner { field1: u32 }
struct Outer(Inner);

impl Outer {
    fn new() -> Self {
        Self(Inner { field1: 42 })
    }

    fn into_inner(self) -> u32 {
        (self.0).field1
    }

    fn into_inner_destructed(self) -> u32 {
        let Outer(Inner { field1: x }) = self;
        x
    }
}"#,
        );

        check_assist(
            convert_tuple_struct_to_named_struct,
            r#"
struct Inner(u32);
struct Outer$0(Inner);

impl Outer {
    fn new() -> Self {
        Self(Inner(42))
    }

    fn into_inner(self) -> u32 {
        (self.0).0
    }

    fn into_inner_destructed(self) -> u32 {
        let Outer(Inner(x)) = self;
        x
    }
}"#,
            r#"
struct Inner(u32);
struct Outer { field1: Inner }

impl Outer {
    fn new() -> Self {
        Self { field1: Inner(42) }
    }

    fn into_inner(self) -> u32 {
        (self.field1).0
    }

    fn into_inner_destructed(self) -> u32 {
        let Outer { field1: Inner(x) } = self;
        x
    }
}"#,
        );
    }

    #[test]
    fn convert_struct_with_multi_file_references() {
        check_assist(
            convert_tuple_struct_to_named_struct,
            r#"
//- /main.rs
struct Inner;
struct A$0(Inner);

mod foo;

//- /foo.rs
use crate::{A, Inner};
fn f() {
    let a = A(Inner);
}
"#,
            r#"
//- /main.rs
struct Inner;
struct A { field1: Inner }

mod foo;

//- /foo.rs
use crate::{A, Inner};
fn f() {
    let a = A { field1: Inner };
}
"#,
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
