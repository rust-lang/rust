use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: undeclared-label
pub(crate) fn undeclared_label(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::UndeclaredLabel,
) -> Diagnostic {
    let name = &d.name;
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcHardError("undeclared-label"),
        format!("use of undeclared label `{}`", name.display(ctx.sema.db)),
        d.node.clone().map(|it| it.into()),
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
    fn smoke_test() {
        check_diagnostics(
            r#"
fn foo() {
    break 'a;
  //^^^^^^^^ error: break outside of loop
        //^^ error: use of undeclared label `'a`
    continue 'a;
  //^^^^^^^^^^^ error: continue outside of loop
           //^^ error: use of undeclared label `'a`
}
"#,
        );
    }

    #[test]
    fn for_loop() {
        check_diagnostics(
            r#"
//- minicore: iterator
fn foo() {
    'xxx: for _ in unknown {
        'yyy: for _ in unknown {
            break 'xxx;
            continue 'yyy;
            break 'zzz;
                //^^^^ error: use of undeclared label `'zzz`
        }
        continue 'xxx;
        continue 'yyy;
               //^^^^ error: use of undeclared label `'yyy`
        break 'xxx;
        break 'yyy;
            //^^^^ error: use of undeclared label `'yyy`
    }
}
"#,
        );
    }

    #[test]
    fn try_operator_desugar_works() {
        check_diagnostics(
            r#"
//- minicore: option, try
fn foo() {
    None?;
}
"#,
        );
        check_diagnostics(
            r#"
//- minicore: option, try, future
async fn foo() {
    None?;
}
"#,
        );
        check_diagnostics(
            r#"
//- minicore: option, try, future, fn
async fn foo() {
    || None?;
}
"#,
        );
    }
}
