use either::Either;
use ide_db::defs::Definition;
use itertools::Itertools;
use syntax::{
    ast::{self, AstNode, HasGenericParams, HasVisibility},
    match_ast, SyntaxKind, SyntaxNode,
};

use crate::{assist_context::SourceChangeBuilder, AssistContext, AssistId, AssistKind, Assists};

// Assist: convert_named_struct_to_tuple_struct
//
// Converts struct with named fields to tuple struct, and analogously for enum variants with named
// fields.
//
// ```
// struct Point$0 { x: f32, y: f32 }
//
// impl Point {
//     pub fn new(x: f32, y: f32) -> Self {
//         Point { x, y }
//     }
//
//     pub fn x(&self) -> f32 {
//         self.x
//     }
//
//     pub fn y(&self) -> f32 {
//         self.y
//     }
// }
// ```
// ->
// ```
// struct Point(f32, f32);
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
pub(crate) fn convert_named_struct_to_tuple_struct(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
) -> Option<()> {
    let strukt = ctx.find_node_at_offset::<Either<ast::Struct, ast::Variant>>()?;
    let field_list = strukt.as_ref().either(|s| s.field_list(), |v| v.field_list())?;
    let record_fields = match field_list {
        ast::FieldList::RecordFieldList(it) => it,
        ast::FieldList::TupleFieldList(_) => return None,
    };
    let strukt_def = match &strukt {
        Either::Left(s) => Either::Left(ctx.sema.to_def(s)?),
        Either::Right(v) => Either::Right(ctx.sema.to_def(v)?),
    };
    let target = strukt.as_ref().either(|s| s.syntax(), |v| v.syntax()).text_range();

    acc.add(
        AssistId("convert_named_struct_to_tuple_struct", AssistKind::RefactorRewrite),
        "Convert to tuple struct",
        target,
        |edit| {
            edit_field_references(ctx, edit, record_fields.fields());
            edit_struct_references(ctx, edit, strukt_def);
            edit_struct_def(ctx, edit, &strukt, record_fields);
        },
    )
}

fn edit_struct_def(
    ctx: &AssistContext<'_>,
    edit: &mut SourceChangeBuilder,
    strukt: &Either<ast::Struct, ast::Variant>,
    record_fields: ast::RecordFieldList,
) {
    let tuple_fields = record_fields
        .fields()
        .filter_map(|f| Some(ast::make::tuple_field(f.visibility(), f.ty()?)));
    let tuple_fields = ast::make::tuple_field_list(tuple_fields);
    let record_fields_text_range = record_fields.syntax().text_range();

    edit.edit_file(ctx.file_id());
    edit.replace(record_fields_text_range, tuple_fields.syntax().text());

    if let Either::Left(strukt) = strukt {
        if let Some(w) = strukt.where_clause() {
            let mut where_clause = w.to_string();
            if where_clause.ends_with(',') {
                where_clause.pop();
            }
            where_clause.push(';');

            edit.delete(w.syntax().text_range());
            edit.insert(record_fields_text_range.end(), ast::make::tokens::single_newline().text());
            edit.insert(record_fields_text_range.end(), where_clause);
            edit.insert(record_fields_text_range.end(), ast::make::tokens::single_newline().text());

            if let Some(tok) = strukt
                .generic_param_list()
                .and_then(|l| l.r_angle_token())
                .and_then(|tok| tok.next_token())
                .filter(|tok| tok.kind() == SyntaxKind::WHITESPACE)
            {
                edit.delete(tok.text_range());
            }
        } else {
            edit.insert(record_fields_text_range.end(), ";");
        }
    }

    if let Some(tok) = record_fields
        .l_curly_token()
        .and_then(|tok| tok.prev_token())
        .filter(|tok| tok.kind() == SyntaxKind::WHITESPACE)
    {
        edit.delete(tok.text_range())
    }
}

fn edit_struct_references(
    ctx: &AssistContext<'_>,
    edit: &mut SourceChangeBuilder,
    strukt: Either<hir::Struct, hir::Variant>,
) {
    let strukt_def = match strukt {
        Either::Left(s) => Definition::Adt(hir::Adt::Struct(s)),
        Either::Right(v) => Definition::Variant(v),
    };
    let usages = strukt_def.usages(&ctx.sema).include_self_refs().all();

    let edit_node = |edit: &mut SourceChangeBuilder, node: SyntaxNode| -> Option<()> {
        match_ast! {
            match node {
                ast::RecordPat(record_struct_pat) => {
                    edit.replace(
                        record_struct_pat.syntax().text_range(),
                        ast::make::tuple_struct_pat(
                            record_struct_pat.path()?,
                            record_struct_pat
                                .record_pat_field_list()?
                                .fields()
                                .filter_map(|pat| pat.pat())
                        )
                        .to_string()
                    );
                },
                ast::RecordExpr(record_expr) => {
                    let path = record_expr.path()?;
                    let args = record_expr
                        .record_expr_field_list()?
                        .fields()
                        .filter_map(|f| f.expr())
                        .join(", ");

                    edit.replace(record_expr.syntax().text_range(), format!("{path}({args})"));
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
    ctx: &AssistContext<'_>,
    edit: &mut SourceChangeBuilder,
    fields: impl Iterator<Item = ast::RecordField>,
) {
    for (index, field) in fields.enumerate() {
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
                    // Only edit the field reference if it's part of a `.field` access
                    if name_ref.syntax().parent().and_then(ast::FieldExpr::cast).is_some() {
                        edit.replace(name_ref.syntax().text_range(), index.to_string());
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn not_applicable_other_than_record_struct() {
        check_assist_not_applicable(convert_named_struct_to_tuple_struct, r#"struct Foo$0(u32)"#);
        check_assist_not_applicable(convert_named_struct_to_tuple_struct, r#"struct Foo$0;"#);
    }

    #[test]
    fn convert_simple_struct() {
        check_assist(
            convert_named_struct_to_tuple_struct,
            r#"
struct Inner;
struct A$0 { inner: Inner }

impl A {
    fn new(inner: Inner) -> A {
        A { inner }
    }

    fn new_with_default() -> A {
        A::new(Inner)
    }

    fn into_inner(self) -> Inner {
        self.inner
    }
}"#,
            r#"
struct Inner;
struct A(Inner);

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
        );
    }

    #[test]
    fn convert_struct_referenced_via_self_kw() {
        check_assist(
            convert_named_struct_to_tuple_struct,
            r#"
struct Inner;
struct A$0 { inner: Inner }

impl A {
    fn new(inner: Inner) -> Self {
        Self { inner }
    }

    fn new_with_default() -> Self {
        Self::new(Inner)
    }

    fn into_inner(self) -> Inner {
        self.inner
    }
}"#,
            r#"
struct Inner;
struct A(Inner);

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
        );
    }

    #[test]
    fn convert_destructured_struct() {
        check_assist(
            convert_named_struct_to_tuple_struct,
            r#"
struct Inner;
struct A$0 { inner: Inner }

impl A {
    fn into_inner(self) -> Inner {
        let A { inner: a } = self;
        a
    }

    fn into_inner_via_self(self) -> Inner {
        let Self { inner } = self;
        inner
    }
}"#,
            r#"
struct Inner;
struct A(Inner);

impl A {
    fn into_inner(self) -> Inner {
        let A(a) = self;
        a
    }

    fn into_inner_via_self(self) -> Inner {
        let Self(inner) = self;
        inner
    }
}"#,
        );
    }

    #[test]
    fn convert_struct_with_visibility() {
        check_assist(
            convert_named_struct_to_tuple_struct,
            r#"
struct A$0 {
    pub first: u32,
    pub(crate) second: u64
}

impl A {
    fn new() -> A {
        A { first: 42, second: 42 }
    }

    fn into_first(self) -> u32 {
        self.first
    }

    fn into_second(self) -> u64 {
        self.second
    }
}"#,
            r#"
struct A(pub u32, pub(crate) u64);

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
        );
    }

    #[test]
    fn convert_struct_with_wrapped_references() {
        check_assist(
            convert_named_struct_to_tuple_struct,
            r#"
struct Inner$0 { uint: u32 }
struct Outer { inner: Inner }

impl Outer {
    fn new() -> Self {
        Self { inner: Inner { uint: 42 } }
    }

    fn into_inner(self) -> u32 {
        self.inner.uint
    }

    fn into_inner_destructed(self) -> u32 {
        let Outer { inner: Inner { uint: x } } = self;
        x
    }
}"#,
            r#"
struct Inner(u32);
struct Outer { inner: Inner }

impl Outer {
    fn new() -> Self {
        Self { inner: Inner(42) }
    }

    fn into_inner(self) -> u32 {
        self.inner.0
    }

    fn into_inner_destructed(self) -> u32 {
        let Outer { inner: Inner(x) } = self;
        x
    }
}"#,
        );

        check_assist(
            convert_named_struct_to_tuple_struct,
            r#"
struct Inner { uint: u32 }
struct Outer$0 { inner: Inner }

impl Outer {
    fn new() -> Self {
        Self { inner: Inner { uint: 42 } }
    }

    fn into_inner(self) -> u32 {
        self.inner.uint
    }

    fn into_inner_destructed(self) -> u32 {
        let Outer { inner: Inner { uint: x } } = self;
        x
    }
}"#,
            r#"
struct Inner { uint: u32 }
struct Outer(Inner);

impl Outer {
    fn new() -> Self {
        Self(Inner { uint: 42 })
    }

    fn into_inner(self) -> u32 {
        self.0.uint
    }

    fn into_inner_destructed(self) -> u32 {
        let Outer(Inner { uint: x }) = self;
        x
    }
}"#,
        );
    }

    #[test]
    fn convert_struct_with_multi_file_references() {
        check_assist(
            convert_named_struct_to_tuple_struct,
            r#"
//- /main.rs
struct Inner;
struct A$0 { inner: Inner }

mod foo;

//- /foo.rs
use crate::{A, Inner};
fn f() {
    let a = A { inner: Inner };
}
"#,
            r#"
//- /main.rs
struct Inner;
struct A(Inner);

mod foo;

//- /foo.rs
use crate::{A, Inner};
fn f() {
    let a = A(Inner);
}
"#,
        );
    }

    #[test]
    fn convert_struct_with_where_clause() {
        check_assist(
            convert_named_struct_to_tuple_struct,
            r#"
struct Wrap$0<T>
where
    T: Display,
{ field1: T }
"#,
            r#"
struct Wrap<T>(T)
where
    T: Display;

"#,
        );
    }

    #[test]
    fn not_applicable_other_than_record_variant() {
        check_assist_not_applicable(
            convert_named_struct_to_tuple_struct,
            r#"enum Enum { Variant$0(usize) };"#,
        );
        check_assist_not_applicable(
            convert_named_struct_to_tuple_struct,
            r#"enum Enum { Variant$0 }"#,
        );
    }

    #[test]
    fn convert_simple_variant() {
        check_assist(
            convert_named_struct_to_tuple_struct,
            r#"
enum A {
    $0Variant { field1: usize },
}

impl A {
    fn new(value: usize) -> A {
        A::Variant { field1: value }
    }

    fn new_with_default() -> A {
        A::new(Default::default())
    }

    fn value(self) -> usize {
        match self {
            A::Variant { field1: value } => value,
        }
    }
}"#,
            r#"
enum A {
    Variant(usize),
}

impl A {
    fn new(value: usize) -> A {
        A::Variant(value)
    }

    fn new_with_default() -> A {
        A::new(Default::default())
    }

    fn value(self) -> usize {
        match self {
            A::Variant(value) => value,
        }
    }
}"#,
        );
    }

    #[test]
    fn convert_variant_referenced_via_self_kw() {
        check_assist(
            convert_named_struct_to_tuple_struct,
            r#"
enum A {
    $0Variant { field1: usize },
}

impl A {
    fn new(value: usize) -> A {
        Self::Variant { field1: value }
    }

    fn new_with_default() -> A {
        Self::new(Default::default())
    }

    fn value(self) -> usize {
        match self {
            Self::Variant { field1: value } => value,
        }
    }
}"#,
            r#"
enum A {
    Variant(usize),
}

impl A {
    fn new(value: usize) -> A {
        Self::Variant(value)
    }

    fn new_with_default() -> A {
        Self::new(Default::default())
    }

    fn value(self) -> usize {
        match self {
            Self::Variant(value) => value,
        }
    }
}"#,
        );
    }

    #[test]
    fn convert_destructured_variant() {
        check_assist(
            convert_named_struct_to_tuple_struct,
            r#"
enum A {
    $0Variant { field1: usize },
}

impl A {
    fn into_inner(self) -> usize {
        let A::Variant { field1: first } = self;
        first
    }

    fn into_inner_via_self(self) -> usize {
        let Self::Variant { field1: first } = self;
        first
    }
}"#,
            r#"
enum A {
    Variant(usize),
}

impl A {
    fn into_inner(self) -> usize {
        let A::Variant(first) = self;
        first
    }

    fn into_inner_via_self(self) -> usize {
        let Self::Variant(first) = self;
        first
    }
}"#,
        );
    }

    #[test]
    fn convert_variant_with_wrapped_references() {
        check_assist(
            convert_named_struct_to_tuple_struct,
            r#"
enum Inner {
    $0Variant { field1: usize },
}
enum Outer {
    Variant(Inner),
}

impl Outer {
    fn new() -> Self {
        Self::Variant(Inner::Variant { field1: 42 })
    }

    fn into_inner_destructed(self) -> u32 {
        let Outer::Variant(Inner::Variant { field1: x }) = self;
        x
    }
}"#,
            r#"
enum Inner {
    Variant(usize),
}
enum Outer {
    Variant(Inner),
}

impl Outer {
    fn new() -> Self {
        Self::Variant(Inner::Variant(42))
    }

    fn into_inner_destructed(self) -> u32 {
        let Outer::Variant(Inner::Variant(x)) = self;
        x
    }
}"#,
        );

        check_assist(
            convert_named_struct_to_tuple_struct,
            r#"
enum Inner {
    Variant(usize),
}
enum Outer {
    $0Variant { field1: Inner },
}

impl Outer {
    fn new() -> Self {
        Self::Variant { field1: Inner::Variant(42) }
    }

    fn into_inner_destructed(self) -> u32 {
        let Outer::Variant { field1: Inner::Variant(x) } = self;
        x
    }
}"#,
            r#"
enum Inner {
    Variant(usize),
}
enum Outer {
    Variant(Inner),
}

impl Outer {
    fn new() -> Self {
        Self::Variant(Inner::Variant(42))
    }

    fn into_inner_destructed(self) -> u32 {
        let Outer::Variant(Inner::Variant(x)) = self;
        x
    }
}"#,
        );
    }

    #[test]
    fn convert_variant_with_multi_file_references() {
        check_assist(
            convert_named_struct_to_tuple_struct,
            r#"
//- /main.rs
struct Inner;
enum A {
    $0Variant { field1: Inner },
}

mod foo;

//- /foo.rs
use crate::{A, Inner};
fn f() {
    let a = A::Variant { field1: Inner };
}
"#,
            r#"
//- /main.rs
struct Inner;
enum A {
    Variant(Inner),
}

mod foo;

//- /foo.rs
use crate::{A, Inner};
fn f() {
    let a = A::Variant(Inner);
}
"#,
        );
    }

    #[test]
    fn convert_directly_used_variant() {
        check_assist(
            convert_named_struct_to_tuple_struct,
            r#"
//- /main.rs
struct Inner;
enum A {
    $0Variant { field1: Inner },
}

mod foo;

//- /foo.rs
use crate::{A::Variant, Inner};
fn f() {
    let a = Variant { field1: Inner };
}
"#,
            r#"
//- /main.rs
struct Inner;
enum A {
    Variant(Inner),
}

mod foo;

//- /foo.rs
use crate::{A::Variant, Inner};
fn f() {
    let a = Variant(Inner);
}
"#,
        );
    }
}
