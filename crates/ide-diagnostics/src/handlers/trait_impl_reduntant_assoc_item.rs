use hir::InFile;
use itertools::Itertools;
use syntax::{ast, AstNode};

use crate::{adjusted_display_range, Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: trait-impl-reduntant-assoc_item
//
// Diagnoses reduntant trait items in a trait impl.
pub(crate) fn trait_impl_reduntant_assoc_item(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::TraitImplReduntantAssocItems,
) -> Diagnostic {
    let reduntant = d.reduntant.iter().format_with(", ", |(name, item), f| {
        f(&match *item {
            hir::AssocItem::Function(_) => "`fn ",
            hir::AssocItem::Const(_) => "`const ",
            hir::AssocItem::TypeAlias(_) => "`type ",
        })?;
        f(&name.display(ctx.sema.db))?;
        f(&"`")
    });
    Diagnostic::new(
        DiagnosticCode::RustcHardError("E0407"),
        format!("{reduntant} is not a member of trait"),
        adjusted_display_range::<ast::Impl>(
            ctx,
            InFile { file_id: d.file_id, value: d.impl_.syntax_node_ptr() },
            &|impl_| impl_.trait_().map(|t| t.syntax().text_range()),
        ),
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
    fn boo();
}
struct Foo;
impl Marker for Foo {
   //^^^^^^ error: `type T`, `const FLAG`, `fn bar` is not a member of trait
    type T = i32;
    const FLAG: bool = false;
    fn bar() {}
    fn boo() {}
}
            "#,
        )
    }
}
