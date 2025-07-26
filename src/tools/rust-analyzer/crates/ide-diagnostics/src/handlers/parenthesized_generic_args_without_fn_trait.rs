use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: parenthesized-generic-args-without-fn-trait
//
// This diagnostic is shown when a `Fn`-trait-style generic parameters (`Trait(A, B) -> C`)
// was used on non-`Fn` trait/type.
pub(crate) fn parenthesized_generic_args_without_fn_trait(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::ParenthesizedGenericArgsWithoutFnTrait,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0214"),
        "parenthesized type parameters may only be used with a `Fn` trait",
        d.args.map(Into::into),
    )
    .stable()
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn fn_traits_work() {
        check_diagnostics(
            r#"
//- minicore: async_fn, fn
fn foo<
    A: Fn(),
    B: FnMut() -> i32,
    C: FnOnce(&str, bool),
    D: AsyncFn::(u32) -> u32,
    E: AsyncFnMut(),
    F: AsyncFnOnce() -> bool,
>() {}
        "#,
        );
    }

    #[test]
    fn non_fn_trait() {
        check_diagnostics(
            r#"
struct Struct<T>(T);
enum Enum<T> { EnumVariant(T) }
type TypeAlias<T> = bool;

type Foo = TypeAlias() -> bool;
                 // ^^ error: parenthesized type parameters may only be used with a `Fn` trait

fn foo(_a: Struct(i32)) {
              // ^^^^^ error: parenthesized type parameters may only be used with a `Fn` trait
    let _ = <Enum::(u32)>::EnumVariant(0);
              // ^^^^^^^ error: parenthesized type parameters may only be used with a `Fn` trait
}
        "#,
        );
    }
}
