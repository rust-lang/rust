use hir::{db::ExpandDatabase, HasSource, HirDisplay, Semantics};
use ide_db::{base_db::FileId, source_change::SourceChange, RootDatabase};
use syntax::{
    ast::{self, edit::IndentLevel, make},
    AstNode,
};
use text_edit::TextEdit;

use crate::{fix, Assist, Diagnostic, DiagnosticsContext};

// Diagnostic: no-such-field
//
// This diagnostic is triggered if created structure does not have field provided in record.
pub(crate) fn no_such_field(ctx: &DiagnosticsContext<'_>, d: &hir::NoSuchField) -> Diagnostic {
    Diagnostic::new(
        "no-such-field",
        "no such field",
        ctx.sema.diagnostics_display_range(d.field.clone().map(|it| it.into())).range,
    )
    .with_fixes(fixes(ctx, d))
}

fn fixes(ctx: &DiagnosticsContext<'_>, d: &hir::NoSuchField) -> Option<Vec<Assist>> {
    let root = ctx.sema.db.parse_or_expand(d.field.file_id);
    missing_record_expr_field_fixes(
        &ctx.sema,
        d.field.file_id.original_file(ctx.sema.db),
        &d.field.value.to_node(&root),
    )
}

fn missing_record_expr_field_fixes(
    sema: &Semantics<'_, RootDatabase>,
    usage_file_id: FileId,
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
        def_file_id,
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
    fn no_such_field_diagnostics() {
        check_diagnostics(
            r#"
struct S { foo: i32, bar: () }
impl S {
    fn new() -> S {
        S {
      //^ 💡 error: missing structure fields:
      //|    - bar
            foo: 92,
            baz: 62,
          //^^^^^^^ 💡 error: no such field
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
}
