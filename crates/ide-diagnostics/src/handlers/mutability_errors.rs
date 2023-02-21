use crate::{Diagnostic, DiagnosticsContext, Severity};

// Diagnostic: need-mut
//
// This diagnostic is triggered on mutating an immutable variable.
pub(crate) fn need_mut(ctx: &DiagnosticsContext<'_>, d: &hir::NeedMut) -> Diagnostic {
    Diagnostic::new(
        "need-mut",
        format!("cannot mutate immutable variable `{}`", d.local.name(ctx.sema.db)),
        ctx.sema.diagnostics_display_range(d.span.clone()).range,
    )
}

// Diagnostic: unused-mut
//
// This diagnostic is triggered when a mutable variable isn't actually mutated.
pub(crate) fn unused_mut(ctx: &DiagnosticsContext<'_>, d: &hir::UnusedMut) -> Diagnostic {
    Diagnostic::new(
        "unused-mut",
        "remove this `mut`",
        ctx.sema.diagnostics_display_range(d.local.primary_source(ctx.sema.db).syntax_ptr()).range,
    )
    .severity(Severity::WeakWarning)
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn unused_mut_simple() {
        check_diagnostics(
            r#"
fn f(_: i32) {}
fn main() {
    let mut x = 2;
      //^^^^^ weak: remove this `mut`
    f(x);
}
"#,
        );
    }

    #[test]
    fn no_false_positive_simple() {
        check_diagnostics(
            r#"
fn f(_: i32) {}
fn main() {
    let x = 2;
    f(x);
}
"#,
        );
        check_diagnostics(
            r#"
fn f(_: i32) {}
fn main() {
    let mut x = 2;
    x = 5;
    f(x);
}
"#,
        );
    }

    #[test]
    fn field_mutate() {
        check_diagnostics(
            r#"
fn f(_: i32) {}
fn main() {
    let mut x = (2, 7);
      //^^^^^ weak: remove this `mut`
    f(x.1);
}
"#,
        );
        check_diagnostics(
            r#"
fn f(_: i32) {}
fn main() {
    let mut x = (2, 7);
    x.0 = 5;
    f(x.1);
}
"#,
        );
        check_diagnostics(
            r#"
fn f(_: i32) {}
fn main() {
    let x = (2, 7);
    x.0 = 5;
  //^^^^^^^ error: cannot mutate immutable variable `x`
    f(x.1);
}
"#,
        );
    }

    #[test]
    fn mutable_reference() {
        check_diagnostics(
            r#"
fn main() {
    let mut x = &mut 2;
      //^^^^^ weak: remove this `mut`
    *x = 5;
}
"#,
        );
        check_diagnostics(
            r#"
fn main() {
    let x = 2;
    &mut x;
  //^^^^^^ error: cannot mutate immutable variable `x`
}
"#,
        );
        check_diagnostics(
            r#"
fn main() {
    let x_own = 2;
    let ref mut x_ref = x_own;
      //^^^^^^^^^^^^^ error: cannot mutate immutable variable `x_own`
}
"#,
        );
        check_diagnostics(
            r#"
struct Foo;
impl Foo {
    fn method(&mut self, x: i32) {}
}
fn main() {
    let x = Foo;
    x.method(2);
  //^ error: cannot mutate immutable variable `x`
}
"#,
        );
    }

    #[test]
    fn match_bindings() {
        check_diagnostics(
            r#"
fn main() {
    match (2, 3) {
        (x, mut y) => {
          //^^^^^ weak: remove this `mut`
            x = 7;
          //^^^^^ error: cannot mutate immutable variable `x`
        }
    }
}
"#,
        );
    }

    #[test]
    fn mutation_in_dead_code() {
        // This one is interesting. Dead code is not represented at all in the MIR, so
        // there would be no mutablility error for locals in dead code. Rustc tries to
        // not emit `unused_mut` in this case, but since it works without `mut`, and
        // special casing it is not trivial, we emit it.
        check_diagnostics(
            r#"
fn main() {
    return;
    let mut x = 2;
      //^^^^^ weak: remove this `mut`
    &mut x;
}
"#,
        );
        check_diagnostics(
            r#"
fn main() {
    loop {}
    let mut x = 2;
      //^^^^^ weak: remove this `mut`
    &mut x;
}
"#,
        );
        check_diagnostics(
            r#"
enum X {}
fn g() -> X {
    loop {}
}
fn f() -> ! {
    loop {}
}
fn main(b: bool) {
    if b {
        f();
    } else {
        g();
    }
    let mut x = 2;
      //^^^^^ weak: remove this `mut`
    &mut x;
}
"#,
        );
        check_diagnostics(
            r#"
fn main(b: bool) {
    if b {
        loop {}
    } else {
        return;
    }
    let mut x = 2;
      //^^^^^ weak: remove this `mut`
    &mut x;
}
"#,
        );
    }

    #[test]
    fn initialization_is_not_mutation() {
        check_diagnostics(
            r#"
fn f(_: i32) {}
fn main() {
    let mut x;
      //^^^^^ weak: remove this `mut`
    x = 5;
    f(x);
}
"#,
        );
        check_diagnostics(
            r#"
fn f(_: i32) {}
fn main(b: bool) {
    let mut x;
      //^^^^^ weak: remove this `mut`
    if b {
        x = 1;
    } else {
        x = 3;
    }
    f(x);
}
"#,
        );
        check_diagnostics(
            r#"
fn f(_: i32) {}
fn main(b: bool) {
    let x;
    if b {
        x = 1;
    }
    x = 3;
  //^^^^^ error: cannot mutate immutable variable `x`
    f(x);
}
"#,
        );
        check_diagnostics(
            r#"
fn f(_: i32) {}
fn main() {
    let x;
    loop {
        x = 1;
      //^^^^^ error: cannot mutate immutable variable `x`
        f(x);
    }
}
"#,
        );
        check_diagnostics(
            r#"
fn f(_: i32) {}
fn main() {
    loop {
        let mut x = 1;
          //^^^^^ weak: remove this `mut`
        f(x);
        if let mut y = 2 {
             //^^^^^ weak: remove this `mut`
            f(y);
        }
        match 3 {
            mut z => f(z),
          //^^^^^ weak: remove this `mut`
        }
    }
}
"#,
        );
    }
}
