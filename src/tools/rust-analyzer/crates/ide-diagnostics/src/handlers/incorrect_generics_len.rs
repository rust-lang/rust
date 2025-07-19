use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};
use hir::IncorrectGenericsLenKind;

// Diagnostic: incorrect-generics-len
//
// This diagnostic is triggered if the number of generic arguments does not match their declaration.
pub(crate) fn incorrect_generics_len(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::IncorrectGenericsLen,
) -> Diagnostic {
    let owner_description = d.def.description();
    let expected = d.expected;
    let provided = d.provided;
    let kind_description = match d.kind {
        IncorrectGenericsLenKind::Lifetimes => "lifetime",
        IncorrectGenericsLenKind::TypesAndConsts => "generic",
    };
    let message = format!(
        "this {owner_description} takes {expected} {kind_description} argument{} \
            but {provided} {kind_description} argument{} {} supplied",
        if expected == 1 { "" } else { "s" },
        if provided == 1 { "" } else { "s" },
        if provided == 1 { "was" } else { "were" },
    );
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("E0107"),
        message,
        d.generics_or_segment.map(Into::into),
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn partially_specified_generics() {
        check_diagnostics(
            r#"
struct Bar<T, U>(T, U);

fn foo() {
    let _ = Bar::<()>;
            // ^^^^^^ error: this struct takes 2 generic arguments but 1 generic argument was supplied
}

        "#,
        );
    }

    #[test]
    fn enum_variant() {
        check_diagnostics(
            r#"
enum Enum<T, U> {
    Variant(T, U),
}

fn foo() {
    let _ = Enum::<()>::Variant;
             // ^^^^^^ error: this enum takes 2 generic arguments but 1 generic argument was supplied
    let _ = Enum::Variant::<()>;
                      // ^^^^^^ error: this enum takes 2 generic arguments but 1 generic argument was supplied
}

        "#,
        );
    }

    #[test]
    fn lifetimes() {
        check_diagnostics(
            r#"
struct Foo<'a, 'b>(&'a &'b ());

fn foo(Foo(_): Foo) -> Foo {
    let _: Foo = Foo(&&());
    let _: Foo::<> = Foo::<>(&&());
    let _: Foo::<'static>
           // ^^^^^^^^^^^ error: this struct takes 2 lifetime arguments but 1 lifetime argument was supplied
                          = Foo::<'static>(&&());
                            // ^^^^^^^^^^^ error: this struct takes 2 lifetime arguments but 1 lifetime argument was supplied
    |_: Foo| -> Foo {loop{}};

    loop {}
}

        "#,
        );
    }

    #[test]
    fn no_error_for_elided_lifetimes() {
        check_diagnostics(
            r#"
struct Foo<'a>(&'a ());

fn foo(_v: &()) -> Foo { loop {} }
        "#,
        );
    }

    #[test]
    fn errs_for_elided_lifetimes_if_lifetimes_are_explicitly_provided() {
        check_diagnostics(
            r#"
struct Foo<'a, 'b>(&'a &'b ());

fn foo(_v: Foo<'_>
           // ^^^^ error: this struct takes 2 lifetime arguments but 1 lifetime argument was supplied
) -> Foo<'static> { loop {} }
     // ^^^^^^^^^ error: this struct takes 2 lifetime arguments but 1 lifetime argument was supplied
        "#,
        );
    }

    #[test]
    fn types_and_consts() {
        check_diagnostics(
            r#"
struct Foo<'a, T>(&'a T);
fn foo(_v: Foo) {}
        // ^^^ error: this struct takes 1 generic argument but 0 generic arguments were supplied

struct Bar<T, const N: usize>(T);
fn bar() {
    let _ = Bar::<()>;
            // ^^^^^^ error: this struct takes 2 generic arguments but 1 generic argument was supplied
}
        "#,
        );
    }

    #[test]
    fn respects_defaults() {
        check_diagnostics(
            r#"
struct Foo<T = (), const N: usize = 0>(T);
fn foo(_v: Foo) {}

struct Bar<T, const N: usize = 0>(T);
fn bar(_v: Bar<()>) {}
        "#,
        );
    }

    #[test]
    fn constant() {
        check_diagnostics(
            r#"
const CONST: i32 = 0;
fn baz() {
    let _ = CONST::<()>;
              // ^^^^^^ error: this constant takes 0 generic arguments but 1 generic argument was supplied
}
        "#,
        );
    }

    #[test]
    fn assoc_type() {
        check_diagnostics(
            r#"
trait Trait {
    type Assoc;
}

fn foo<T: Trait<Assoc<i32> = bool>>() {}
                  // ^^^^^ error: this type alias takes 0 generic arguments but 1 generic argument was supplied
        "#,
        );
    }

    #[test]
    fn regression_19669() {
        check_diagnostics(
            r#"
//- minicore: from
fn main() {
    let _: i32 = Into::into(0);
}
"#,
        );
    }

    #[test]
    fn generic_assoc_type_infer_lifetime_in_expr_position() {
        check_diagnostics(
            r#"
//- minicore: sized
struct Player;

struct Foo<'c, C> {
    _v: &'c C,
}
trait WithSignals: Sized {
    type SignalCollection<'c, C>;
    fn __signals_from_external(&self) -> Self::SignalCollection<'_, Self>;
}
impl WithSignals for Player {
    type SignalCollection<'c, C> = Foo<'c, C>;
    fn __signals_from_external(&self) -> Self::SignalCollection<'_, Self> {
        Self::SignalCollection { _v: self }
    }
}
        "#,
        );
    }
}
