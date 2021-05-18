use hir::{db::AstDatabase, diagnostics::NoSuchField, HasSource, HirDisplay, Semantics};
use ide_db::{base_db::FileId, source_change::SourceChange, RootDatabase};
use syntax::{
    ast::{self, edit::IndentLevel, make},
    AstNode,
};
use text_edit::TextEdit;

use crate::{
    diagnostics::{fix, DiagnosticWithFixes},
    Assist, AssistResolveStrategy,
};
impl DiagnosticWithFixes for NoSuchField {
    fn fixes(
        &self,
        sema: &Semantics<RootDatabase>,
        _resolve: &AssistResolveStrategy,
    ) -> Option<Vec<Assist>> {
        let root = sema.db.parse_or_expand(self.file)?;
        missing_record_expr_field_fixes(
            &sema,
            self.file.original_file(sema.db),
            &self.field.to_node(&root),
        )
    }
}

fn missing_record_expr_field_fixes(
    sema: &Semantics<RootDatabase>,
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

    let new_field_type = sema.type_of_expr(&record_expr_field.expr()?)?;
    if new_field_type.is_unknown() {
        return None;
    }
    let new_field = make::record_field(
        None,
        make::name(&record_expr_field.field_name()?.text()),
        make::ty(&new_field_type.display_source_code(sema.db, module.into()).ok()?),
    );

    let last_field = record_fields.fields().last()?;
    let last_field_syntax = last_field.syntax();
    let indent = IndentLevel::from_node(last_field_syntax);

    let mut new_field = new_field.to_string();
    if usage_file_id != def_file_id {
        new_field = format!("pub(crate) {}", new_field);
    }
    new_field = format!("\n{}{}", indent, new_field);

    let needs_comma = !last_field_syntax.to_string().ends_with(',');
    if needs_comma {
        new_field = format!(",{}", new_field);
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
    use crate::diagnostics::tests::check_fix;

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
struct Foo {
    bar: i32
}
"#,
            r#"
struct Foo {
    bar: i32,
    pub(crate) baz: bool
}
"#,
        )
    }
}
