use either::Either;
use ide_db::{defs::Definition, search::FileReference};
use itertools::Itertools;
use syntax::{
    SyntaxKind,
    ast::{self, AstNode, HasAttrs, HasGenericParams, HasVisibility},
    match_ast, ted,
};

use crate::{AssistContext, AssistId, Assists, assist_context::SourceChangeBuilder};

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
    // XXX: We don't currently provide this assist for struct definitions inside macros, but if we
    // are to lift this limitation, don't forget to make `edit_struct_def()` consider macro files
    // too.
    let name = ctx.find_node_at_offset::<ast::Name>()?;
    let strukt = name.syntax().parent().and_then(<Either<ast::Struct, ast::Variant>>::cast)?;
    let field_list = strukt.as_ref().either(|s| s.field_list(), |v| v.field_list())?;
    let record_fields = match field_list {
        ast::FieldList::RecordFieldList(it) => it,
        ast::FieldList::TupleFieldList(_) => return None,
    };
    let strukt_def = match &strukt {
        Either::Left(s) => Either::Left(ctx.sema.to_def(s)?),
        Either::Right(v) => Either::Right(ctx.sema.to_def(v)?),
    };

    acc.add(
        AssistId::refactor_rewrite("convert_named_struct_to_tuple_struct"),
        "Convert to tuple struct",
        strukt.syntax().text_range(),
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
    // Note that we don't need to consider macro files in this function because this is
    // currently not triggered for struct definitions inside macro calls.
    let tuple_fields = record_fields.fields().filter_map(|f| {
        let field = ast::make::tuple_field(f.visibility(), f.ty()?).clone_for_update();
        ted::insert_all(
            ted::Position::first_child_of(field.syntax()),
            f.attrs().map(|attr| attr.syntax().clone_subtree().clone_for_update().into()).collect(),
        );
        Some(field)
    });
    let tuple_fields = ast::make::tuple_field_list(tuple_fields);
    let record_fields_text_range = record_fields.syntax().text_range();

    edit.edit_file(ctx.vfs_file_id());
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

    for (file_id, refs) in usages {
        edit.edit_file(file_id.file_id(ctx.db()));
        for r in refs {
            process_struct_name_reference(ctx, r, edit);
        }
    }
}

fn process_struct_name_reference(
    ctx: &AssistContext<'_>,
    r: FileReference,
    edit: &mut SourceChangeBuilder,
) -> Option<()> {
    // First check if it's the last semgnet of a path that directly belongs to a record
    // expression/pattern.
    let name_ref = r.name.as_name_ref()?;
    let path_segment = name_ref.syntax().parent().and_then(ast::PathSegment::cast)?;
    // A `PathSegment` always belongs to a `Path`, so there's at least one `Path` at this point.
    let full_path =
        path_segment.syntax().parent()?.ancestors().map_while(ast::Path::cast).last()?;

    if full_path.segment()?.name_ref()? != *name_ref {
        // `name_ref` isn't the last segment of the path, so `full_path` doesn't point to the
        // struct we want to edit.
        return None;
    }

    let parent = full_path.syntax().parent()?;
    match_ast! {
        match parent {
            ast::RecordPat(record_struct_pat) => {
                // When we failed to get the original range for the whole struct expression node,
                // we can't provide any reasonable edit. Leave it untouched.
                let file_range = ctx.sema.original_range_opt(record_struct_pat.syntax())?;
                edit.replace(
                    file_range.range,
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
                // When we failed to get the original range for the whole struct pattern node,
                // we can't provide any reasonable edit. Leave it untouched.
                let file_range = ctx.sema.original_range_opt(record_expr.syntax())?;
                let path = record_expr.path()?;
                let args = record_expr
                    .record_expr_field_list()?
                    .fields()
                    .filter_map(|f| f.expr())
                    .join(", ");

                edit.replace(file_range.range, format!("{path}({args})"));
            },
            _ => {}
        }
    }

    Some(())
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
            edit.edit_file(file_id.file_id(ctx.db()));
            for r in refs {
                if let Some(name_ref) = r.name.as_name_ref() {
                    // Only edit the field reference if it's part of a `.field` access
                    if name_ref.syntax().parent().and_then(ast::FieldExpr::cast).is_some() {
                        edit.replace(r.range, index.to_string());
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

    #[test]
    fn field_access_inside_macro_call() {
        check_assist(
            convert_named_struct_to_tuple_struct,
            r#"
struct $0Struct {
    inner: i32,
}

macro_rules! id {
    ($e:expr) => { $e }
}

fn test(c: Struct) {
    id!(c.inner);
}
"#,
            r#"
struct Struct(i32);

macro_rules! id {
    ($e:expr) => { $e }
}

fn test(c: Struct) {
    id!(c.0);
}
"#,
        )
    }

    #[test]
    fn struct_usage_inside_macro_call() {
        check_assist(
            convert_named_struct_to_tuple_struct,
            r#"
macro_rules! id {
    ($($t:tt)*) => { $($t)* }
}

struct $0Struct {
    inner: i32,
}

fn test() {
    id! {
        let s = Struct {
            inner: 42,
        };
        let Struct { inner: value } = s;
        let Struct { inner } = s;
    }
}
"#,
            r#"
macro_rules! id {
    ($($t:tt)*) => { $($t)* }
}

struct Struct(i32);

fn test() {
    id! {
        let s = Struct(42);
        let Struct(value) = s;
        let Struct(inner) = s;
    }
}
"#,
        );
    }

    #[test]
    fn struct_name_ref_may_not_be_part_of_struct_expr_or_struct_pat() {
        check_assist(
            convert_named_struct_to_tuple_struct,
            r#"
struct $0Struct {
    inner: i32,
}
struct Outer<T> {
    value: T,
}
fn foo<T>() -> T { loop {} }

fn test() {
    Outer {
        value: foo::<Struct>();
    }
}

trait HasAssoc {
    type Assoc;
    fn test();
}
impl HasAssoc for Struct {
    type Assoc = Outer<i32>;
    fn test() {
        let a = Self::Assoc {
            value: 42,
        };
        let Self::Assoc { value } = a;
    }
}
"#,
            r#"
struct Struct(i32);
struct Outer<T> {
    value: T,
}
fn foo<T>() -> T { loop {} }

fn test() {
    Outer {
        value: foo::<Struct>();
    }
}

trait HasAssoc {
    type Assoc;
    fn test();
}
impl HasAssoc for Struct {
    type Assoc = Outer<i32>;
    fn test() {
        let a = Self::Assoc {
            value: 42,
        };
        let Self::Assoc { value } = a;
    }
}
"#,
        );
    }

    #[test]
    fn fields_with_attrs() {
        check_assist(
            convert_named_struct_to_tuple_struct,
            r#"
pub struct $0Foo {
    #[my_custom_attr]
    value: u32,
}
"#,
            r#"
pub struct Foo(#[my_custom_attr]
u32);
"#,
        );
    }
}
