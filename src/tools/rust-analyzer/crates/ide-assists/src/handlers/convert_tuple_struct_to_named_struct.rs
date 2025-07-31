use either::Either;
use hir::FileRangeWrapper;
use ide_db::defs::{Definition, NameRefClass};
use std::ops::RangeInclusive;
use syntax::{
    SyntaxElement, SyntaxKind, SyntaxNode, T, TextSize,
    ast::{
        self, AstNode, HasAttrs, HasGenericParams, HasVisibility, syntax_factory::SyntaxFactory,
    },
    match_ast,
    syntax_editor::{Element, Position, SyntaxEditor},
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
    let strukt_or_variant = ctx
        .find_node_at_offset::<ast::Struct>()
        .map(Either::Left)
        .or_else(|| ctx.find_node_at_offset::<ast::Variant>().map(Either::Right))?;
    let field_list = strukt_or_variant.as_ref().either(|s| s.field_list(), |v| v.field_list())?;

    if ctx.offset() > field_list.syntax().text_range().start() {
        // Assist could be distracting after the braces
        return None;
    }

    let tuple_fields = match field_list {
        ast::FieldList::TupleFieldList(it) => it,
        ast::FieldList::RecordFieldList(_) => return None,
    };
    let strukt_def = match &strukt_or_variant {
        Either::Left(s) => Either::Left(ctx.sema.to_def(s)?),
        Either::Right(v) => Either::Right(ctx.sema.to_def(v)?),
    };
    let target = strukt_or_variant.as_ref().either(|s| s.syntax(), |v| v.syntax()).text_range();
    let syntax = strukt_or_variant.as_ref().either(|s| s.syntax(), |v| v.syntax());
    acc.add(
        AssistId::refactor_rewrite("convert_tuple_struct_to_named_struct"),
        "Convert to named struct",
        target,
        |edit| {
            let names = generate_names(tuple_fields.fields());
            edit_field_references(ctx, edit, tuple_fields.fields(), &names);
            let mut editor = edit.make_editor(syntax);
            edit_struct_references(ctx, edit, strukt_def, &names);
            edit_struct_def(&mut editor, &strukt_or_variant, tuple_fields, names);
            edit.add_file_edits(ctx.vfs_file_id(), editor);
        },
    )
}

fn edit_struct_def(
    editor: &mut SyntaxEditor,
    strukt: &Either<ast::Struct, ast::Variant>,
    tuple_fields: ast::TupleFieldList,
    names: Vec<ast::Name>,
) {
    let record_fields = tuple_fields.fields().zip(names).filter_map(|(f, name)| {
        let field = ast::make::record_field(f.visibility(), name, f.ty()?);
        let mut field_editor = SyntaxEditor::new(field.syntax().clone());
        field_editor.insert_all(
            Position::first_child_of(field.syntax()),
            f.attrs().map(|attr| attr.syntax().clone_subtree().clone_for_update().into()).collect(),
        );
        ast::RecordField::cast(field_editor.finish().new_root().clone())
    });
    let make = SyntaxFactory::without_mappings();
    let record_fields = make.record_field_list(record_fields);
    let tuple_fields_before = Position::before(tuple_fields.syntax());

    if let Either::Left(strukt) = strukt {
        if let Some(w) = strukt.where_clause() {
            editor.delete(w.syntax());
            let mut insert_element = Vec::new();
            insert_element.push(ast::make::tokens::single_newline().syntax_element());
            insert_element.push(w.syntax().clone_for_update().syntax_element());
            if w.syntax().last_token().is_none_or(|t| t.kind() != SyntaxKind::COMMA) {
                insert_element.push(ast::make::token(T![,]).into());
            }
            insert_element.push(ast::make::tokens::single_newline().syntax_element());
            editor.insert_all(tuple_fields_before, insert_element);
        } else {
            editor.insert(tuple_fields_before, ast::make::tokens::single_space());
        }
        if let Some(t) = strukt.semicolon_token() {
            editor.delete(t);
        }
    } else {
        editor.insert(tuple_fields_before, ast::make::tokens::single_space());
    }

    editor.replace(tuple_fields.syntax(), record_fields.syntax());
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

    let edit_node = |node: SyntaxNode| -> Option<SyntaxNode> {
        let make = SyntaxFactory::without_mappings();
        match_ast! {
            match node {
                ast::TupleStructPat(tuple_struct_pat) => {
                    Some(make.record_pat_with_fields(
                        tuple_struct_pat.path()?,
                        ast::make::record_pat_field_list(tuple_struct_pat.fields().zip(names).map(
                            |(pat, name)| {
                                ast::make::record_pat_field(
                                    ast::make::name_ref(&name.to_string()),
                                    pat,
                                )
                            },
                        ), None),
                    ).syntax().clone())
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
                    Some(
                        make.record_expr(
                            path,
                            ast::make::record_expr_field_list(arg_list.args().zip(names).map(
                                |(expr, name)| {
                                    ast::make::record_expr_field(
                                        ast::make::name_ref(&name.to_string()),
                                        Some(expr),
                                    )
                                },
                            )),
                        ).syntax().clone()
                    )
                },
                _ => None,
            }
        }
    };

    for (file_id, refs) in usages {
        let source = ctx.sema.parse(file_id);
        let source = source.syntax();

        let mut editor = edit.make_editor(source);
        for r in refs.iter().rev() {
            if let Some((old_node, new_node)) = r
                .name
                .syntax()
                .ancestors()
                .find_map(|node| Some((node.clone(), edit_node(node.clone())?)))
            {
                if let Some(old_node) = ctx.sema.original_syntax_node_rooted(&old_node) {
                    editor.replace(old_node, new_node);
                } else {
                    let FileRangeWrapper { file_id: _, range } = ctx.sema.original_range(&old_node);
                    let parent = source.covering_element(range);
                    match parent {
                        SyntaxElement::Token(token) => {
                            editor.replace(token, new_node.syntax_element());
                        }
                        SyntaxElement::Node(parent_node) => {
                            // replace the part of macro
                            // ```
                            // foo!(a, Test::A(0));
                            //     ^^^^^^^^^^^^^^^ // parent_node
                            //         ^^^^^^^^^^  // replace_range
                            // ```
                            let start = parent_node
                                .children_with_tokens()
                                .find(|t| t.text_range().contains(range.start()));
                            let end = parent_node
                                .children_with_tokens()
                                .find(|t| t.text_range().contains(range.end() - TextSize::new(1)));
                            if let (Some(start), Some(end)) = (start, end) {
                                let replace_range = RangeInclusive::new(start, end);
                                editor.replace_all(replace_range, vec![new_node.into()]);
                            }
                        }
                    }
                }
            }
        }
        edit.add_file_edits(file_id.file_id(ctx.db()), editor);
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
            let source = ctx.sema.parse(file_id);
            let source = source.syntax();
            let mut editor = edit.make_editor(source);
            for r in refs {
                if let Some(name_ref) = r.name.as_name_ref()
                    && let Some(original) = ctx.sema.original_ast_node(name_ref.clone())
                {
                    editor.replace(original.syntax(), name.syntax());
                }
            }
            edit.add_file_edits(file_id.file_id(ctx.db()), editor);
        }
    }
}

fn generate_names(fields: impl Iterator<Item = ast::TupleField>) -> Vec<ast::Name> {
    let make = SyntaxFactory::without_mappings();
    fields
        .enumerate()
        .map(|(i, _)| {
            let idx = i + 1;
            make.name(&format!("field{idx}"))
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
    fn convert_simple_struct_cursor_on_struct_keyword() {
        check_assist(
            convert_tuple_struct_to_named_struct,
            r#"
struct Inner;
struct$0 A(Inner);

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
    fn convert_simple_struct_cursor_on_visibility_keyword() {
        check_assist(
            convert_tuple_struct_to_named_struct,
            r#"
struct Inner;
pub$0 struct A(Inner);

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
pub struct A { field1: Inner }

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
pub struct Foo { #[my_custom_attr]field1: u32 }
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
