use hir::HirDisplay;

use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: unimplemented-trait
//
// This diagnostic is triggered when rust-analyzer cannot infer some type.
pub(crate) fn unimplemented_trait<'db>(
    ctx: &DiagnosticsContext<'_, 'db>,
    d: &hir::UnimplementedTrait<'db>,
) -> Diagnostic {
    let message = match &d.root_trait_predicate {
        Some(root_predicate) if *root_predicate != d.trait_predicate => format!(
            "the trait bound `{}` is not satisfied\n\
            required by the bound `{}`\n",
            d.trait_predicate.display(ctx.db(), ctx.display_target),
            root_predicate.display(ctx.db(), ctx.display_target),
        ),
        _ => format!(
            "the trait bound `{}` is not satisfied",
            d.trait_predicate.display(ctx.db(), ctx.display_target),
        ),
    };
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0277"),
        message,
        d.span.map(Into::into),
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn smoke_test() {
        check_diagnostics(
            r#"
trait Trait {}
impl<T: Trait, const N: usize> Trait for [T; N] {}
fn foo(_v: impl Trait) {}
fn bar() {
    foo(1);
 // ^^^ error: the trait bound `i32: Trait` is not satisfied
    foo([1]);
 // ^^^ error: the trait bound `i32: Trait` is not satisfied
   // | required by the bound `[i32; 1]: Trait`
}
        "#,
        );
    }

    #[test]
    fn async_closure_does_not_trigger() {
        check_diagnostics(
            r#"
//- minicore: async_fn
fn spawn_in<AsyncFn>(_f: AsyncFn)
where
    AsyncFn: AsyncFnOnce(),
{
}

fn foo() {
    spawn_in(async move || {});
}

        "#,
        );
    }
}
