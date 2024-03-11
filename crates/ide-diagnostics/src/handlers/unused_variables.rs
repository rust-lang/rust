use ide_db::{
    assists::{Assist, AssistId, AssistKind},
    base_db::FileRange,
    label::Label,
    source_change::SourceChange,
};
use text_edit::TextEdit;

use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: unused-variables
//
// This diagnostic is triggered when a local variable is not used.
pub(crate) fn unused_variables(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::UnusedVariable,
) -> Diagnostic {
    let ast = d.local.primary_source(ctx.sema.db).syntax_ptr();
    let diagnostic_range = ctx.sema.diagnostics_display_range(ast);
    let var_name = d.local.primary_source(ctx.sema.db).syntax().to_string();
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::RustcLint("unused_variables"),
        "unused variable",
        ast,
    )
    .with_fixes(fixes(&var_name, diagnostic_range, ast.file_id.is_macro()))
    .experimental()
}

fn fixes(var_name: &String, diagnostic_range: FileRange, is_in_marco: bool) -> Option<Vec<Assist>> {
    if is_in_marco {
        return None;
    }
    Some(vec![Assist {
        id: AssistId("unscore_unused_variable_name", AssistKind::QuickFix),
        label: Label::new(format!("Rename unused {} to _{}", var_name, var_name)),
        group: None,
        target: diagnostic_range.range,
        source_change: Some(SourceChange::from_text_edit(
            diagnostic_range.file_id,
            TextEdit::replace(diagnostic_range.range, format!("_{}", var_name)),
        )),
        trigger_signature_help: false,
    }])
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_diagnostics, check_fix, check_no_fix};

    #[test]
    fn unused_variables_simple() {
        check_diagnostics(
            r#"
//- minicore: fn
struct Foo { f1: i32, f2: i64 }

fn f(kkk: i32) {}
   //^^^ 💡 warn: unused variable
fn main() {
    let a = 2;
      //^ 💡 warn: unused variable
    let b = 5;
    // note: `unused variable` implies `unused mut`, so we should not emit both at the same time.
    let mut c = f(b);
      //^^^^^ 💡 warn: unused variable
    let (d, e) = (3, 5);
       //^ 💡 warn: unused variable
    let _ = e;
    let f1 = 2;
    let f2 = 5;
    let f = Foo { f1, f2 };
    match f {
        Foo { f1, f2 } => {
            //^^ 💡 warn: unused variable
            _ = f2;
        }
    }
    let g = false;
    if g {}
    let h: fn() -> i32 = || 2;
    let i = h();
      //^ 💡 warn: unused variable
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
                      //^ 💡 warn: unused variable
    fn ref_self(&self, u: i32) {}
                     //^ 💡 warn: unused variable
    fn ref_mut_self(&mut self, u: i32) {}
                             //^ 💡 warn: unused variable
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
      //^ 💡 error: unused variable
}
"#,
        );
    }

    #[test]
    fn fix_unused_variable() {
        check_fix(
            r#"
fn main() {
    let x$0 = 2;
}
"#,
            r#"
fn main() {
    let _x = 2;
}
"#,
        );

        check_fix(
            r#"
fn main() {
    let ($0d, _e) = (3, 5);
}
"#,
            r#"
fn main() {
    let (_d, _e) = (3, 5);
}
"#,
        );

        check_fix(
            r#"
struct Foo { f1: i32, f2: i64 }
fn main() {
    let f = Foo { f1: 0, f2: 0 };
    match f {
        Foo { f1$0, f2 } => {
            _ = f2;
        }
    }
}
"#,
            r#"
struct Foo { f1: i32, f2: i64 }
fn main() {
    let f = Foo { f1: 0, f2: 0 };
    match f {
        Foo { _f1, f2 } => {
            _ = f2;
        }
    }
}
"#,
        );
    }

    #[test]
    fn no_fix_for_marco() {
        check_no_fix(
            r#"
macro_rules! my_macro {
    () => {
        let x = 3;
    };
}

fn main() {
    $0my_macro!();
}
"#,
        );
    }
}
