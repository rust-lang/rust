use either::Either;
use ide_db::defs::{Definition, NameRefClass};
use syntax::{
    SyntaxKind, SyntaxNode,
    ast::{self, AstNode, HasAttrs, HasGenericParams, HasVisibility},
    match_ast, ted,
};

use crate::{AssistContext, AssistId, Assists, assist_context::SourceChangeBuilder};

// Assist: convert_tuple_struct_to_named_struct
//
// Converts tuple struct to struct with named fields, and analogously for tuple enum variants.
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
    ctx: &AssistContext<'_>,
) -> Option<()> {
    let name = ctx.find_node_at_offset::<ast::Name>()?;
    let strukt = name.syntax().parent().and_then(<Either<ast::Struct, ast::Variant>>::cast)?;
    let field_list = strukt.as_ref().either(|s| s.field_list(), |v| v.field_list())?;
    let tuple_fields = match field_list {
        ast::FieldList::TupleFieldList(it) => it,
        ast::FieldList::RecordFieldList(_) => return None,
    };
    let strukt_def = match &strukt {
        Either::Left(s) => Either::Left(ctx.sema.to_def(s)?),
        Either::Right(v) => Either::Right(ctx.sema.to_def(v)?),
    };
    let target = strukt.as_ref().either(|s| s.syntax(), |v| v.syntax()).text_range();

    acc.add(
        AssistId::refactor_rewrite("convert_tuple_struct_to_named_struct"),
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
    ctx: &AssistContext<'_>,
    edit: &mut SourceChangeBuilder,
    strukt: &Either<ast::Struct, ast::Variant>,
    tuple_fields: ast::TupleFieldList,
    names: Vec<ast::Name>,
) {
    let record_fields = tuple_fields.fields().zip(names).filter_map(|(f, name)| {
        let field = ast::make::record_field(f.visibility(), name, f.ty()?).clone_for_update();
        ted::insert_all(
            ted::Position::first_child_of(field.syntax()),
            f.attrs().map(|attr| attr.syntax().clone_subtree().clone_for_update().into()).collect(),
        );
        Some(field)
    });
    let record_fields = ast::make::record_field_list(record_fields);
    let tuple_fields_text_range = tuple_fields.syntax().text_range();

    edit.edit_file(ctx.vfs_file_id());

    if let Either::Left(strukt) = strukt {
        if let Some(w) = strukt.where_clause() {
            edit.delete(w.syntax().text_range());
            edit.insert(
                tuple_fields_text_range.start(),
                ast::make::tokens::single_newline().text(),
            );
            edit.insert(tuple_fields_text_range.start(), w.syntax().text());
            if w.syntax().last_token().is_none_or(|t| t.kind() != SyntaxKind::COMMA) {
                edit.insert(tuple_fields_text_range.start(), ",");
            }
            edit.insert(
                tuple_fields_text_range.start(),
                ast::make::tokens::single_newline().text(),
            );
        } else {
            edit.insert(tuple_fields_text_range.start(), ast::make::tokens::single_space().text());
        }
        if let Some(t) = strukt.semicolon_token() {
            edit.delete(t.text_range());
        }
    } else {
        edit.insert(tuple_fields_text_range.start(), ast::make::tokens::single_space().text());
    }

    edit.replace(tuple_fields_text_range, record_fields.to_string());
}

fn edit_struct_references(
    ctx: &AssistContext<'_>,
    edit: &mut SourceChangeBuilder,
    strukt: Either<hir::Struct, hir::Variant>,
    names: &[ast::Name],
) {
    let strukt_def = match strukt {
        Either::Left(s) => Definition::Adt(hir::Adt::Struct(s)),
        Either::Right(v) => Definition::Variant(v),
    };
    let usages = strukt_def.usages(&ctx.sema).include_self_refs().all();

    let edit_node = |edit: &mut SourceChangeBuilder, node: SyntaxNode| -> Option<()> {
        match_ast! {
            match node {
                ast::TupleStructPat(tuple_struct_pat) => {
                    let file_range = ctx.sema.original_range_opt(&node)?;
                    edit.edit_file(file_range.file_id.file_id(ctx.db()));
                    edit.replace(
                        file_range.range,
                        ast::make::record_pat_with_fields(
                            tuple_struct_pat.path()?,
                            ast::make::record_pat_field_list(tuple_struct_pat.fields().zip(names).map(
                                |(pat, name)| {
                                    ast::make::record_pat_field(
                                        ast::make::name_ref(&name.to_string()),
                                        pat,
                                    )
                                },
                            ), None),
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
                            Some(NameRefClass::Definition(Definition::SelfType(_), _)) => {},
                            Some(NameRefClass::Definition(def, _)) if def == strukt_def => {},
                            _ => return None,
                        };
                    }

                    let arg_list = call_expr.syntax().descendants().find_map(ast::ArgList::cast)?;

                    edit.replace(
                        ctx.sema.original_range(&node).range,
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
        edit.edit_file(file_id.file_id(ctx.db()));
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
            edit.edit_file(file_id.file_id(ctx.db()));
            for r in refs {
                if let Some(name_ref) = r.name.as_name_ref() {
                    edit.replace(ctx.sema.original_range(name_ref.syntax()).range, name.text());
                }
            }
        }
    }
}

fn generate_names(fields: impl Iterator<Item = ast::TupleField>) -> Vec<ast::Name> {
    fields
        .enumerate()
        .map(|(i, _)| {
            let idx = i + 1;
            ast::make::name(&format!("field{idx}"))
        })
        .collect()
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
    fn convert_in_macro_args() {
        check_assist(
            convert_tuple_struct_to_named_struct,
            r#"
macro_rules! foo {($i:expr) => {$i} }
struct T$0(u8);
fn test() {
    foo!(T(1));
}"#,
            r#"
macro_rules! foo {($i:expr) => {$i} }
struct T { field1: u8 }
fn test() {
    foo!(T { field1: 1 });
}"#,
        );
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
    #[test]
    fn not_applicable_other_than_tuple_variant() {
        check_assist_not_applicable(
            convert_tuple_struct_to_named_struct,
            r#"enum Enum { Variant$0 { value: usize } };"#,
        );
        check_assist_not_applicable(
            convert_tuple_struct_to_named_struct,
            r#"enum Enum { Variant$0 }"#,
        );
    }

    #[test]
    fn convert_variant_in_macro_args() {
        check_assist(
            convert_tuple_struct_to_named_struct,
            r#"
macro_rules! foo {($i:expr) => {$i} }
enum T {
  V$0(u8)
}
fn test() {
    foo!(T::V(1));
}"#,
            r#"
macro_rules! foo {($i:expr) => {$i} }
enum T {
  V { field1: u8 }
}
fn test() {
    foo!(T::V { field1: 1 });
}"#,
        );
    }

    #[test]
    fn convert_simple_variant() {
        check_assist(
            convert_tuple_struct_to_named_struct,
            r#"
enum A {
    $0Variant(usize),
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
            r#"
enum A {
    Variant { field1: usize },
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
        );
    }

    #[test]
    fn convert_variant_referenced_via_self_kw() {
        check_assist(
            convert_tuple_struct_to_named_struct,
            r#"
enum A {
    $0Variant(usize),
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
            r#"
enum A {
    Variant { field1: usize },
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
        );
    }

    #[test]
    fn convert_destructured_variant() {
        check_assist(
            convert_tuple_struct_to_named_struct,
            r#"
enum A {
    $0Variant(usize),
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
            r#"
enum A {
    Variant { field1: usize },
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
        );
    }

    #[test]
    fn convert_variant_with_wrapped_references() {
        check_assist(
            convert_tuple_struct_to_named_struct,
            r#"
enum Inner {
    $0Variant(usize),
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
            r#"
enum Inner {
    Variant { field1: usize },
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
        );

        check_assist(
            convert_tuple_struct_to_named_struct,
            r#"
enum Inner {
    Variant(usize),
}
enum Outer {
    $0Variant(Inner),
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
            r#"
enum Inner {
    Variant(usize),
}
enum Outer {
    Variant { field1: Inner },
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
        );
    }

    #[test]
    fn convert_variant_with_multi_file_references() {
        check_assist(
            convert_tuple_struct_to_named_struct,
            r#"
//- /main.rs
struct Inner;
enum A {
    $0Variant(Inner),
}

mod foo;

//- /foo.rs
use crate::{A, Inner};
fn f() {
    let a = A::Variant(Inner);
}
"#,
            r#"
//- /main.rs
struct Inner;
enum A {
    Variant { field1: Inner },
}

mod foo;

//- /foo.rs
use crate::{A, Inner};
fn f() {
    let a = A::Variant { field1: Inner };
}
"#,
        );
    }

    #[test]
    fn convert_directly_used_variant() {
        check_assist(
            convert_tuple_struct_to_named_struct,
            r#"
//- /main.rs
struct Inner;
enum A {
    $0Variant(Inner),
}

mod foo;

//- /foo.rs
use crate::{A::Variant, Inner};
fn f() {
    let a = Variant(Inner);
}
"#,
            r#"
//- /main.rs
struct Inner;
enum A {
    Variant { field1: Inner },
}

mod foo;

//- /foo.rs
use crate::{A::Variant, Inner};
fn f() {
    let a = Variant { field1: Inner };
}
"#,
        );
    }

    #[test]
    fn where_clause_with_trailing_comma() {
        check_assist(
            convert_tuple_struct_to_named_struct,
            r#"
trait Foo {}

struct Bar$0<T>(pub T)
where
    T: Foo,;
"#,
            r#"
trait Foo {}

struct Bar<T>
where
    T: Foo,
{ pub field1: T }

"#,
        );
    }

    #[test]
    fn fields_with_attrs() {
        check_assist(
            convert_tuple_struct_to_named_struct,
            r#"
pub struct $0Foo(#[my_custom_attr] u32);
"#,
            r#"
pub struct Foo { #[my_custom_attr]
field1: u32 }
"#,
        );
    }

    #[test]
    fn convert_in_macro_pattern_args() {
        check_assist(
            convert_tuple_struct_to_named_struct,
            r#"
macro_rules! foo {
    ($expression:expr, $pattern:pat) => {
        match $expression {
            $pattern => true,
            _ => false
        }
    };
}
enum Expr {
    A$0(usize),
}
fn main() {
    let e = Expr::A(0);
    foo!(e, Expr::A(0));
}
"#,
            r#"
macro_rules! foo {
    ($expression:expr, $pattern:pat) => {
        match $expression {
            $pattern => true,
            _ => false
        }
    };
}
enum Expr {
    A { field1: usize },
}
fn main() {
    let e = Expr::A { field1: 0 };
    foo!(e, Expr::A { field1: 0 });
}
"#,
        );
    }

    #[test]
    fn convert_in_multi_file_macro_pattern_args() {
        check_assist(
            convert_tuple_struct_to_named_struct,
            r#"
//- /main.rs
mod foo;

enum Test {
    A$0(i32)
}

//- /foo.rs
use crate::Test;

macro_rules! foo {
    ($expression:expr, $pattern:pat) => {
        match $expression {
            $pattern => true,
            _ => false
        }
    };
}

fn foo() {
    let a = Test::A(0);
    foo!(a, Test::A(0));
}
"#,
            r#"
//- /main.rs
mod foo;

enum Test {
    A { field1: i32 }
}

//- /foo.rs
use crate::Test;

macro_rules! foo {
    ($expression:expr, $pattern:pat) => {
        match $expression {
            $pattern => true,
            _ => false
        }
    };
}

fn foo() {
    let a = Test::A { field1: 0 };
    foo!(a, Test::A { field1: 0 });
}
"#,
        );
    }
}
