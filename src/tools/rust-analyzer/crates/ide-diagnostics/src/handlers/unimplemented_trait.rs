use hir::HirDisplay;

use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: unimplemented-trait
//
// This diagnostic is triggered when a trait bound is not satisfied.
pub(crate) fn unimplemented_trait<'db>(
    ctx: &DiagnosticsContext<'_, 'db>,
    d: &hir::UnimplementedTrait<'db>,
) -> Diagnostic {
    let mut message = format!(
        "the trait bound `{}` is not satisfied",
        d.trait_predicate.display(ctx.db(), ctx.display_target),
    );
    for parent_predicate in &d.parent_trait_predicates {
        message.push_str(&format!(
            "\nrequired by the bound `{}`",
            parent_predicate.display(ctx.db(), ctx.display_target),
        ));
    }
    if !d.parent_trait_predicates.is_empty() {
        message.push('\n');
    }
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
    foo([[1]]);
 // ^^^ error: the trait bound `i32: Trait` is not satisfied
   // | required by the bound `[i32; 1]: Trait`
   // | required by the bound `[[i32; 1]; 1]: Trait`
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

    #[test]
    fn for_iterable() {
        check_diagnostics(
            r#"
//- minicore: iterator
fn foo() {
    for _ in () {}
          // ^^ error: the trait bound `(): Iterator` is not satisfied
           // | required by the bound `(): IntoIterator`
}

        "#,
        );
    }

    #[test]
    fn coroutine_non_held_types_with_auto_traits() {
        check_diagnostics(
            r#"
auto trait Send {}

struct NonSend;
impl !Send for NonSend {}

async fn foo() -> NonSend {
    let v = NonSend;
    v
}

fn require_send<T: Send>(_: T) {}

fn baz() {
    require_send(foo());
}
        "#,
        );
    }
}
