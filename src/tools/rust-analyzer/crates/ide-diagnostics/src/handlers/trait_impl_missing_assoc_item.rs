use hir::InFile;
use itertools::Itertools;
use syntax::{ast, AstNode};

use crate::{adjusted_display_range, Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: trait-impl-missing-assoc_item
//
// Diagnoses missing trait items in a trait impl.
pub(crate) fn trait_impl_missing_assoc_item(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::TraitImplMissingAssocItems,
) -> Diagnostic {
    let missing = d.missing.iter().format_with(", ", |(name, item), f| {
        f(&match *item {
            hir::AssocItem::Function(_) => "`fn ",
            hir::AssocItem::Const(_) => "`const ",
            hir::AssocItem::TypeAlias(_) => "`type ",
        })?;
        f(&name.display(ctx.sema.db, ctx.edition))?;
        f(&"`")
    });
    Diagnostic::new(
        DiagnosticCode::RustcHardError("E0046"),
        format!("not all trait items implemented, missing: {missing}"),
        adjusted_display_range::<ast::Impl>(
            ctx,
            InFile { file_id: d.file_id, value: d.impl_ },
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
    const FLAG: bool = false;
}
struct Foo;
impl Marker for Foo {}
            "#,
        )
    }

    #[test]
    fn simple() {
        check_diagnostics(
            r#"
trait Trait {
    const C: ();
    type T;
    fn f();
}

impl Trait for () {
    const C: () = ();
    type T = ();
    fn f() {}
}

impl Trait for () {
   //^^^^^ error: not all trait items implemented, missing: `const C`
    type T = ();
    fn f() {}
}

impl Trait for () {
   //^^^^^ error: not all trait items implemented, missing: `const C`, `type T`, `fn f`
}

"#,
        );
    }

    #[test]
    fn default() {
        check_diagnostics(
            r#"
trait Trait {
    const C: ();
    type T = ();
    fn f() {}
}

impl Trait for () {
    const C: () = ();
    type T = ();
    fn f() {}
}

impl Trait for () {
   //^^^^^ error: not all trait items implemented, missing: `const C`
    type T = ();
    fn f() {}
}

impl Trait for () {
   //^^^^^ error: not all trait items implemented, missing: `const C`
     type T = ();
 }

impl Trait for () {
   //^^^^^ error: not all trait items implemented, missing: `const C`
}

"#,
        );
    }

    #[test]
    fn negative_impl() {
        check_diagnostics(
            r#"
trait Trait {
    fn item();
}

// Negative impls don't require any items (in fact, the forbid providing any)
impl !Trait for () {}
"#,
        )
    }
}
