use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: private-field
//
// This diagnostic is triggered if the accessed field is not visible from the current module.
pub(crate) fn private_field(ctx: &DiagnosticsContext<'_>, d: &hir::PrivateField) -> Diagnostic {
    // FIXME: add quickfix
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0616"),
        format!(
            "field `{}` of `{}` is private",
            d.field.name(ctx.sema.db).display(ctx.sema.db),
            d.field.parent_def(ctx.sema.db).name(ctx.sema.db).display(ctx.sema.db)
        ),
        d.expr.clone().map(|it| it.into()),
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn private_field() {
        check_diagnostics(
            r#"
mod module { pub struct Struct { field: u32 } }
fn main(s: module::Struct) {
    s.field;
  //^^^^^^^ error: field `field` of `Struct` is private
}
"#,
        );
    }

    #[test]
    fn private_tuple_field() {
        check_diagnostics(
            r#"
mod module { pub struct Struct(u32); }
fn main(s: module::Struct) {
    s.0;
  //^^^ error: field `0` of `Struct` is private
}
"#,
        );
    }

    #[test]
    fn private_but_shadowed_in_deref() {
        check_diagnostics(
            r#"
//- minicore: deref
mod module {
    pub struct Struct { field: Inner }
    pub struct Inner { pub field: u32 }
    impl core::ops::Deref for Struct {
        type Target = Inner;
        fn deref(&self) -> &Inner { &self.field }
    }
}
fn main(s: module::Struct) {
    s.field;
}
"#,
        );
    }

    #[test]
    fn block_module_madness() {
        check_diagnostics(
            r#"
fn main() {
    let strukt = {
        use crate as ForceParentBlockDefMap;
        {
            pub struct Struct {
                field: (),
            }
            Struct { field: () }
        }
    };
    strukt.field;
}
"#,
        );
    }
}
