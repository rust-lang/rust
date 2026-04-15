use either::Either;
use ide_db::{defs::Definition, search::FileReference};
use syntax::{
    NodeOrToken, SyntaxKind, SyntaxNode, T,
    algo::next_non_trivia_token,
    ast::{self, AstNode, HasAttrs, HasGenericParams, HasVisibility},
    match_ast,
    syntax_editor::{Element, Position, SyntaxEditor},
};

use crate::{
    AssistContext, AssistId, Assists, assist_context::SourceChangeBuilder, utils::cover_edit_range,
};

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
    let strukt_or_variant = ctx
        .find_node_at_offset::<ast::Struct>()
        .map(Either::Left)
        .or_else(|| ctx.find_node_at_offset::<ast::Variant>().map(Either::Right))?;
    let field_list = strukt_or_variant.as_ref().either(|s| s.field_list(), |v| v.field_list())?;

    if ctx.offset() > field_list.syntax().text_range().start() {
        // Assist could be distracting after the braces
        return None;
    }

    let record_fields = match field_list {
        ast::FieldList::RecordFieldList(it) => it,
        ast::FieldList::TupleFieldList(_) => return None,
    };
    let strukt_def = match &strukt_or_variant {
        Either::Left(s) => Either::Left(ctx.sema.to_def(s)?),
        Either::Right(v) => Either::Right(ctx.sema.to_def(v)?),
    };

    acc.add(
        AssistId::refactor_rewrite("convert_named_struct_to_tuple_struct"),
        "Convert to tuple struct",
        strukt_or_variant.syntax().text_range(),
        |builder| {
            edit_field_references(ctx, builder, record_fields.fields());
            edit_struct_references(ctx, builder, strukt_def);
            edit_struct_def(ctx, builder, &strukt_or_variant, record_fields);
        },
    )
}

fn edit_struct_def(
    ctx: &AssistContext<'_>,
    builder: &mut SourceChangeBuilder,
    strukt: &Either<ast::Struct, ast::Variant>,
    record_fields: ast::RecordFieldList,
) {
    // Note that we don't need to consider macro files in this function because this is
    // currently not triggered for struct definitions inside macro calls.
    let tuple_fields = record_fields.fields().filter_map(|f| {
        let (editor, field) =
            SyntaxEditor::with_ast_node(&ast::make::tuple_field(f.visibility(), f.ty()?));
        editor.insert_all(
            Position::first_child_of(field.syntax()),
            f.attrs().map(|attr| attr.syntax().clone().into()).collect(),
        );
        let field_syntax = editor.finish().new_root().clone();
        let field = ast::TupleField::cast(field_syntax)?;
        Some(field)
    });

    let editor = builder.make_editor(strukt.syntax());
    let make = editor.make();

    let tuple_fields = make.tuple_field_list(tuple_fields);

    let mut elements = vec![tuple_fields.syntax().clone().into()];
    if let Either::Left(strukt) = strukt {
        if let Some(w) = strukt.where_clause() {
            editor.delete(w.syntax());

            elements.extend([
                make.whitespace("\n").into(),
                remove_trailing_comma(w).into(),
                make.token(T![;]).into(),
                make.whitespace("\n").into(),
            ]);

            if let Some(tok) = strukt
                .generic_param_list()
                .and_then(|l| l.r_angle_token())
                .and_then(|tok| tok.next_token())
                .filter(|tok| tok.kind() == SyntaxKind::WHITESPACE)
            {
                editor.delete(tok);
            }
        } else {
            elements.push(make.token(T![;]).into());
        }
    }
    editor.replace_with_many(record_fields.syntax(), elements);

    if let Some(tok) = record_fields
        .l_curly_token()
        .and_then(|tok| tok.prev_token())
        .filter(|tok| tok.kind() == SyntaxKind::WHITESPACE)
    {
        editor.delete(tok)
    }

    builder.add_file_edits(ctx.vfs_file_id(), editor);
}

fn edit_struct_references(
    ctx: &AssistContext<'_>,
    builder: &mut SourceChangeBuilder,
    strukt: Either<hir::Struct, hir::EnumVariant>,
) {
    let strukt_def = match strukt {
        Either::Left(s) => Definition::Adt(hir::Adt::Struct(s)),
        Either::Right(v) => Definition::EnumVariant(v),
    };
    let usages = strukt_def.usages(&ctx.sema).include_self_refs().all();

    for (file_id, refs) in usages {
        let source = ctx.sema.parse(file_id);
        let editor = builder.make_editor(source.syntax());
        for r in refs {
            process_struct_name_reference(ctx, r, &editor, &source);
        }
        builder.add_file_edits(file_id.file_id(ctx.db()), editor);
    }
}

fn process_struct_name_reference(
    ctx: &AssistContext<'_>,
    r: FileReference,
    edit: &SyntaxEditor,
    source: &ast::SourceFile,
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

    // FIXME: Processing RecordPat and RecordExpr for unordered fields, and insert RestPat
    let parent = full_path.syntax().parent()?;
    match_ast! {
        match parent {
            ast::RecordPat(record_struct_pat) => {
                // When we failed to get the original range for the whole struct pattern node,
                // we can't provide any reasonable edit. Leave it untouched.
                record_to_tuple_struct_like(
                    ctx,
                    source,
                    edit,
                    record_struct_pat.record_pat_field_list()?,
                    |it| it.fields().filter_map(|it| it.name_ref()),
                );
            },
            ast::RecordExpr(record_expr) => {
                // When we failed to get the original range for the whole struct expression node,
                // we can't provide any reasonable edit. Leave it untouched.
                record_to_tuple_struct_like(
                    ctx,
                    source,
                    edit,
                    record_expr.record_expr_field_list()?,
                    |it| it.fields().filter_map(|it| it.name_ref()),
                );
            },
            _ => {}
        }
    }

    Some(())
}

fn record_to_tuple_struct_like<T, I>(
    ctx: &AssistContext<'_>,
    source: &ast::SourceFile,
    editor: &SyntaxEditor,
    field_list: T,
    fields: impl FnOnce(&T) -> I,
) -> Option<()>
where
    T: AstNode,
    I: IntoIterator<Item = ast::NameRef>,
{
    let make = editor.make();
    let orig = ctx.sema.original_range_opt(field_list.syntax())?;
    let list_range = cover_edit_range(source.syntax(), orig.range);

    let l_curly = match list_range.start() {
        NodeOrToken::Node(node) => node.first_token()?,
        NodeOrToken::Token(t) => t.clone(),
    };
    let r_curly = match list_range.end() {
        NodeOrToken::Node(node) => node.last_token()?,
        NodeOrToken::Token(t) => t.clone(),
    };

    if l_curly.kind() == T!['{'] {
        delete_whitespace(editor, l_curly.prev_token());
        delete_whitespace(editor, l_curly.next_token());
        editor.replace(l_curly, make.token(T!['(']));
    }
    if r_curly.kind() == T!['}'] {
        delete_whitespace(editor, r_curly.prev_token());
        editor.replace(r_curly, make.token(T![')']));
    }

    for name_ref in fields(&field_list) {
        let Some(orig) = ctx.sema.original_range_opt(name_ref.syntax()) else { continue };
        let name_range = cover_edit_range(source.syntax(), orig.range);

        if let Some(colon) = next_non_trivia_token(name_range.end().clone())
            && colon.kind() == T![:]
        {
            editor.delete(&colon);
            editor.delete_all(name_range);

            if let Some(next) = next_non_trivia_token(colon.clone())
                && next.kind() != T!['}']
            {
                // Avoid overlapping delete whitespace on `{ field: }`
                delete_whitespace(editor, colon.next_token());
            }
        }
    }
    Some(())
}

fn edit_field_references(
    ctx: &AssistContext<'_>,
    builder: &mut SourceChangeBuilder,
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
            let source = ctx.sema.parse(file_id);
            let editor = builder.make_editor(source.syntax());
            let make = editor.make();

            for r in refs {
                if let Some(name_ref) = r.name.as_name_ref() {
                    // Only edit the field reference if it's part of a `.field` access
                    if name_ref.syntax().parent().and_then(ast::FieldExpr::cast).is_some() {
                        editor.replace_all(
                            cover_edit_range(source.syntax(), r.range),
                            vec![make.name_ref(&index.to_string()).syntax().clone().into()],
                        );
                    }
                }
            }

            builder.add_file_edits(file_id.file_id(ctx.db()), editor);
        }
    }
}

fn delete_whitespace(edit: &SyntaxEditor, whitespace: Option<impl Element>) {
    let Some(whitespace) = whitespace else { return };
    let NodeOrToken::Token(token) = whitespace.syntax_element() else { return };

    if token.kind() == SyntaxKind::WHITESPACE && !token.text().contains('\n') {
        edit.delete(token);
    }
}

fn remove_trailing_comma(w: ast::WhereClause) -> SyntaxNode {
    let (editor, w) = SyntaxEditor::new(w.syntax().clone());
    if let Some(last) = w.last_child_or_token()
        && last.kind() == T![,]
    {
        editor.delete(last);
    }
    editor.finish().new_root().clone()
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
    fn convert_simple_struct_cursor_on_struct_keyword() {
        check_assist(
            convert_named_struct_to_tuple_struct,
            r#"
struct Inner;
struct$0 A { inner: Inner }

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
    fn convert_struct_and_rest_pat() {
        check_assist(
            convert_named_struct_to_tuple_struct,
            r#"
struct Inner;
struct A$0 { inner: Inner }
fn foo(A { .. }: A) {}
"#,
            r#"
struct Inner;
struct A(Inner);
fn foo(A(..): A) {}
"#,
        );

        check_assist(
            convert_named_struct_to_tuple_struct,
            r#"
struct Inner;
struct A$0 { inner: Inner, extra: Inner }
fn foo(A { inner, .. }: A) {}
"#,
            r#"
struct Inner;
struct A(Inner, Inner);
fn foo(A(inner, ..): A) {}
"#,
        );
    }

    #[test]
    fn convert_simple_struct_cursor_on_visibility_keyword() {
        check_assist(
            convert_named_struct_to_tuple_struct,
            r#"
struct Inner;
pub$0 struct A { inner: Inner }

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
pub struct A(Inner);

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
    fn convert_constructor_expr_uses_self() {
        // regression test for #21595
        check_assist(
            convert_named_struct_to_tuple_struct,
            r#"
struct $0Foo { field1: u32 }
impl Foo {
    fn clone(&self) -> Self {
        Self { field1: self.field1 }
    }
}"#,
            r#"
struct Foo(u32);
impl Foo {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}"#,
        );

        check_assist(
            convert_named_struct_to_tuple_struct,
            r#"
macro_rules! id {
    ($($t:tt)*) => { $($t)* }
}
struct $0Foo { field1: u32 }
impl Foo {
    fn clone(&self) -> Self {
        id!(Self { field1: self.field1 })
    }
}"#,
            r#"
macro_rules! id {
    ($($t:tt)*) => { $($t)* }
}
struct Foo(u32);
impl Foo {
    fn clone(&self) -> Self {
        id!(Self(self.0))
    }
}"#,
        );
    }

    #[test]
    fn convert_pat_uses_self() {
        // regression test for #21595
        check_assist(
            convert_named_struct_to_tuple_struct,
            r#"
enum Foo {
    $0Value { field: &'static Foo },
    Nil,
}
fn foo(foo: &Foo) {
    if let Foo::Value { field: Foo::Value { field } } = foo {}
}"#,
            r#"
enum Foo {
    Value(&'static Foo),
    Nil,
}
fn foo(foo: &Foo) {
    if let Foo::Value(Foo::Value(field)) = foo {}
}"#,
        );

        check_assist(
            convert_named_struct_to_tuple_struct,
            r#"
macro_rules! id {
    ($($t:tt)*) => { $($t)* }
}
enum Foo {
    $0Value { field: &'static Foo },
    Nil,
}
fn foo(foo: &Foo) {
    if let id!(Foo::Value { field: Foo::Value { field } }) = foo {}
}"#,
            r#"
macro_rules! id {
    ($($t:tt)*) => { $($t)* }
}
enum Foo {
    Value(&'static Foo),
    Nil,
}
fn foo(foo: &Foo) {
    if let id!(Foo::Value(Foo::Value(field))) = foo {}
}"#,
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
        let s = Struct(
            42,
        );
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
pub struct Foo(#[my_custom_attr]u32);
"#,
        );
    }
}
