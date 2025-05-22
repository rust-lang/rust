use either::Either;
use hir::{HasSource, HirDisplay, Semantics, VariantId, db::ExpandDatabase};
use ide_db::text_edit::TextEdit;
use ide_db::{EditionedFileId, RootDatabase, source_change::SourceChange};
use syntax::{
    AstNode,
    ast::{self, edit::IndentLevel, make},
};

use crate::{Assist, Diagnostic, DiagnosticCode, DiagnosticsContext, fix};

// Diagnostic: no-such-field
//
// This diagnostic is triggered if created structure does not have field provided in record.
pub(crate) fn no_such_field(ctx: &DiagnosticsContext<'_>, d: &hir::NoSuchField) -> Diagnostic {
    let node = d.field.map(Into::into);
    if d.private {
        // FIXME: quickfix to add required visibility
        Diagnostic::new_with_syntax_node_ptr(
            ctx,
            DiagnosticCode::RustcHardError("E0451"),
            "field is private",
            node,
        )
        .stable()
    } else {
        Diagnostic::new_with_syntax_node_ptr(
            ctx,
            match d.variant {
                VariantId::EnumVariantId(_) => DiagnosticCode::RustcHardError("E0559"),
                _ => DiagnosticCode::RustcHardError("E0560"),
            },
            "no such field",
            node,
        )
        .stable()
        .with_fixes(fixes(ctx, d))
    }
}

fn fixes(ctx: &DiagnosticsContext<'_>, d: &hir::NoSuchField) -> Option<Vec<Assist>> {
    // FIXME: quickfix for pattern
    let root = ctx.sema.db.parse_or_expand(d.field.file_id);
    match &d.field.value.to_node(&root) {
        Either::Left(node) => missing_record_expr_field_fixes(
            &ctx.sema,
            d.field.file_id.original_file(ctx.sema.db),
            node,
        ),
        _ => None,
    }
}

fn missing_record_expr_field_fixes(
    sema: &Semantics<'_, RootDatabase>,
    usage_file_id: EditionedFileId,
    record_expr_field: &ast::RecordExprField,
) -> Option<Vec<Assist>> {
    let record_lit = ast::RecordExpr::cast(record_expr_field.syntax().parent()?.parent()?)?;
    let def_id = sema.resolve_variant(record_lit)?;
    let module;
    let def_file_id;
    let record_fields = match def_id {
        hir::VariantDef::Struct(s) => {
            module = s.module(sema.db);
            let source = s.source(sema.db)?;
            def_file_id = source.file_id;
            let fields = source.value.field_list()?;
            record_field_list(fields)?
        }
        hir::VariantDef::Union(u) => {
            module = u.module(sema.db);
            let source = u.source(sema.db)?;
            def_file_id = source.file_id;
            source.value.record_field_list()?
        }
        hir::VariantDef::Variant(e) => {
            module = e.module(sema.db);
            let source = e.source(sema.db)?;
            def_file_id = source.file_id;
            let fields = source.value.field_list()?;
            record_field_list(fields)?
        }
    };
    let def_file_id = def_file_id.original_file(sema.db);

    let new_field_type = sema.type_of_expr(&record_expr_field.expr()?)?.adjusted();
    if new_field_type.is_unknown() {
        return None;
    }
    let new_field = make::record_field(
        None,
        make::name(record_expr_field.field_name()?.ident_token()?.text()),
        make::ty(&new_field_type.display_source_code(sema.db, module.into(), true).ok()?),
    );

    let last_field = record_fields.fields().last()?;
    let last_field_syntax = last_field.syntax();
    let indent = IndentLevel::from_node(last_field_syntax);

    let mut new_field = new_field.to_string();
    if usage_file_id != def_file_id {
        new_field = format!("pub(crate) {new_field}");
    }
    new_field = format!("\n{indent}{new_field}");

    let needs_comma = !last_field_syntax.to_string().ends_with(',');
    if needs_comma {
        new_field = format!(",{new_field}");
    }

    let source_change = SourceChange::from_text_edit(
        def_file_id.file_id(sema.db),
        TextEdit::insert(last_field_syntax.text_range().end(), new_field),
    );

    return Some(vec![fix(
        "create_field",
        "Create field",
        source_change,
        record_expr_field.syntax().text_range(),
    )]);

    fn record_field_list(field_def_list: ast::FieldList) -> Option<ast::RecordFieldList> {
        match field_def_list {
            ast::FieldList::RecordFieldList(it) => Some(it),
            ast::FieldList::TupleFieldList(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_diagnostics, check_fix, check_no_fix};

    #[test]
    fn dont_work_for_field_with_disabled_cfg() {
        check_diagnostics(
            r#"
struct Test {
    #[cfg(feature = "hello")]
    test: u32,
    other: u32
}

fn main() {
    let a = Test {
        #[cfg(feature = "hello")]
        test: 1,
        other: 1
    };

    let Test {
        #[cfg(feature = "hello")]
        test,
        mut other,
        ..
    } = a;

    other += 1;
}
"#,
        );
    }

    #[test]
    fn no_such_field_diagnostics() {
        check_diagnostics(
            r#"
struct S { foo: i32, bar: () }
impl S {
    fn new(
        s@S {
        //^ 💡 error: missing structure fields:
        //|    - bar
            foo,
            baz: baz2,
          //^^^^^^^^^ error: no such field
            qux
          //^^^ error: no such field
        }: S
    ) -> S {
        S {
      //^ 💡 error: missing structure fields:
      //|    - bar
            foo,
            baz: baz2,
          //^^^^^^^^^ error: no such field
            qux
          //^^^ error: no such field
        } = s;
        S {
      //^ 💡 error: missing structure fields:
      //|    - bar
            foo: 92,
            baz: 62,
          //^^^^^^^ 💡 error: no such field
            qux
          //^^^ error: no such field
        }
    }
}
"#,
        );
    }
    #[test]
    fn no_such_field_with_feature_flag_diagnostics() {
        check_diagnostics(
            r#"
//- /lib.rs crate:foo cfg:feature=foo
struct MyStruct {
    my_val: usize,
    #[cfg(feature = "foo")]
    bar: bool,
}

impl MyStruct {
    #[cfg(feature = "foo")]
    pub(crate) fn new(my_val: usize, bar: bool) -> Self {
        Self { my_val, bar }
    }
    #[cfg(not(feature = "foo"))]
    pub(crate) fn new(my_val: usize, _bar: bool) -> Self {
        Self { my_val }
    }
}
"#,
        );
    }

    #[test]
    fn no_such_field_enum_with_feature_flag_diagnostics() {
        check_diagnostics(
            r#"
//- /lib.rs crate:foo cfg:feature=foo
enum Foo {
    #[cfg(not(feature = "foo"))]
    Buz,
    #[cfg(feature = "foo")]
    Bar,
    Baz
}

fn test_fn(f: Foo) {
    match f {
        Foo::Bar => {},
        Foo::Baz => {},
    }
}
"#,
        );
    }

    #[test]
    fn no_such_field_with_feature_flag_diagnostics_on_struct_lit() {
        check_diagnostics(
            r#"
//- /lib.rs crate:foo cfg:feature=foo
struct S {
    #[cfg(feature = "foo")]
    foo: u32,
    #[cfg(not(feature = "foo"))]
    bar: u32,
}

impl S {
    #[cfg(feature = "foo")]
    fn new(foo: u32) -> Self {
        Self { foo }
    }
    #[cfg(not(feature = "foo"))]
    fn new(bar: u32) -> Self {
        Self { bar }
    }
    fn new2(bar: u32) -> Self {
        #[cfg(feature = "foo")]
        { Self { foo: bar } }
        #[cfg(not(feature = "foo"))]
        { Self { bar } }
    }
    fn new2(val: u32) -> Self {
        Self {
            #[cfg(feature = "foo")]
            foo: val,
            #[cfg(not(feature = "foo"))]
            bar: val,
        }
    }
}
"#,
        );
    }

    #[test]
    fn no_such_field_with_type_macro() {
        check_diagnostics(
            r#"
macro_rules! Type { () => { u32 }; }
struct Foo { bar: Type![] }

impl Foo {
    fn new() -> Self {
        Foo { bar: 0 }
    }
}
"#,
        );
    }

    #[test]
    fn test_add_field_from_usage() {
        check_fix(
            r"
fn main() {
    Foo { bar: 3, baz$0: false};
}
struct Foo {
    bar: i32
}
",
            r"
fn main() {
    Foo { bar: 3, baz: false};
}
struct Foo {
    bar: i32,
    baz: bool
}
",
        )
    }

    #[test]
    fn test_add_field_in_other_file_from_usage() {
        check_fix(
            r#"
//- /main.rs
mod foo;

fn main() {
    foo::Foo { bar: 3, $0baz: false};
}
//- /foo.rs
pub struct Foo {
    bar: i32
}
"#,
            r#"
pub struct Foo {
    bar: i32,
    pub(crate) baz: bool
}
"#,
        )
    }

    #[test]
    fn test_tuple_field_on_record_struct() {
        check_no_fix(
            r#"
struct Struct {}
fn main() {
    Struct {
        0$0: 0
    }
}
"#,
        )
    }

    #[test]
    fn test_struct_field_private() {
        check_diagnostics(
            r#"
mod m {
    pub struct Struct {
        field: u32,
        field2: u32,
    }
}
fn f(s@m::Struct {
    field: f,
  //^^^^^^^^ error: field is private
    field2
  //^^^^^^ error: field is private
}: m::Struct) {
    // assignee expression
    m::Struct {
        field: 0,
      //^^^^^^^^ error: field is private
        field2
      //^^^^^^ error: field is private
    } = s;
    m::Struct {
        field: 0,
      //^^^^^^^^ error: field is private
        field2
      //^^^^^^ error: field is private
    };
}
"#,
        )
    }

    #[test]
    fn editions_between_macros() {
        check_diagnostics(
            r#"
//- /edition2015.rs crate:edition2015 edition:2015
#[macro_export]
macro_rules! pass_expr_thorough {
    ($e:expr) => { $e };
}

//- /edition2018.rs crate:edition2018 deps:edition2015 edition:2018
async fn bar() {}
async fn foo() {
    edition2015::pass_expr_thorough!(bar().await);
}
        "#,
        );
        check_diagnostics(
            r#"
//- /edition2018.rs crate:edition2018 edition:2018
pub async fn bar() {}
#[macro_export]
macro_rules! make_await {
    () => { async { $crate::bar().await }; };
}

//- /edition2015.rs crate:edition2015 deps:edition2018 edition:2015
fn foo() {
    edition2018::make_await!();
}
        "#,
        );
    }
}
