use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: unused-variables
//
// This diagnostic is triggered when a local variable is not used.
pub(crate) fn unused_variables(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::UnusedVariable,
) -> Diagnostic {
    let ast = d.local.primary_source(ctx.sema.db).syntax_ptr();
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcLint("unused_variables"),
        "unused variable",
        ast,
    )
    .experimental()
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn unused_variables_simple() {
        check_diagnostics(
            r#"
//- minicore: fn
struct Foo { f1: i32, f2: i64 }

fn f(kkk: i32) {}
   //^^^ warn: unused variable
fn main() {
    let a = 2;
      //^ warn: unused variable
    let b = 5;
    // note: `unused variable` implies `unused mut`, so we should not emit both at the same time.
    let mut c = f(b);
      //^^^^^ warn: unused variable
    let (d, e) = (3, 5);
       //^ warn: unused variable
    let _ = e;
    let f1 = 2;
    let f2 = 5;
    let f = Foo { f1, f2 };
    match f {
        Foo { f1, f2 } => {
            //^^ warn: unused variable
            _ = f2;
        }
    }
    let g = false;
    if g {}
    let h: fn() -> i32 = || 2;
    let i = h();
      //^ warn: unused variable
}
"#,
        );
    }

    #[test]
    fn unused_self() {
        check_diagnostics(
            r#"
struct S {
}
impl S {
    fn owned_self(self, u: i32) {}
                      //^ warn: unused variable
    fn ref_self(&self, u: i32) {}
                     //^ warn: unused variable
    fn ref_mut_self(&mut self, u: i32) {}
                             //^ warn: unused variable
    fn owned_mut_self(mut self) {}
                    //^^^^^^^^ 💡 warn: variable does not need to be mutable

}
"#,
        );
    }

    #[test]
    fn allow_unused_variables_for_identifiers_starting_with_underline() {
        check_diagnostics(
            r#"
fn main() {
    let _x = 2;
}
"#,
        );
    }

    #[test]
    fn respect_lint_attributes_for_unused_variables() {
        check_diagnostics(
            r#"
fn main() {
    #[allow(unused_variables)]
    let x = 2;
}

#[deny(unused)]
fn main2() {
    let x = 2;
      //^ error: unused variable
}
"#,
        );
    }
}
