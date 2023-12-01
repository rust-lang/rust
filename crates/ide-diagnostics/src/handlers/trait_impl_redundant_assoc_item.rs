use hir::{db::ExpandDatabase, Const, Function, HasSource, TypeAlias};

use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: trait-impl-redundant-assoc_item
//
// Diagnoses redundant trait items in a trait impl.
pub(crate) fn trait_impl_redundant_assoc_item(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::TraitImplRedundantAssocItems,
) -> Diagnostic {
    let name = d.assoc_item.0.clone();
    let assoc_item = d.assoc_item.1;
    let db = ctx.sema.db;

    let range = db.parse_or_expand(d.file_id).text_range();
    let trait_name = d.trait_.name(db).to_smol_str();

    let (redundant_item_name, diagnostic_range) = match assoc_item {
        hir::AssocItem::Function(id) => (
            format!("`fn {}`", name.display(db)),
            Function::from(id).source(db).map(|it| it.syntax().value.text_range()).unwrap_or(range),
        ),
        hir::AssocItem::Const(id) => (
            format!("`const {}`", name.display(db)),
            Const::from(id).source(db).map(|it| it.syntax().value.text_range()).unwrap_or(range),
        ),
        hir::AssocItem::TypeAlias(id) => (
            format!("`type {}`", name.display(db)),
            TypeAlias::from(id)
                .source(db)
                .map(|it| it.syntax().value.text_range())
                .unwrap_or(range),
        ),
    };

    Diagnostic::new(
        DiagnosticCode::RustcHardError("E0407"),
        format!("{redundant_item_name} is not a member of trait `{trait_name}`"),
        diagnostic_range,
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn trait_with_default_value() {
        check_diagnostics(
            r#"
trait Marker {
    const FLAG: bool = false;
    fn boo();
    fn foo () {}
}
struct Foo;
impl Marker for Foo {
    type T = i32;
  //^^^^^^^^^^^^^ error: `type T` is not a member of trait `Marker`

    const FLAG: bool = true;

    fn bar() {}
  //^^^^^^^^^^^ error: `fn bar` is not a member of trait `Marker`

    fn boo() {}
}
            "#,
        )
    }
}
