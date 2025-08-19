use hir::Name;
use ide_db::text_edit::TextEdit;
use ide_db::{
    FileRange, RootDatabase,
    assists::{Assist, AssistId},
    label::Label,
    source_change::SourceChange,
};
use syntax::{Edition, TextRange};

use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: unused-variables
//
// This diagnostic is triggered when a local variable is not used.
pub(crate) fn unused_variables(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::UnusedVariable,
) -> Option<Diagnostic> {
    let ast = d.local.primary_source(ctx.sema.db).syntax_ptr();
    if ast.file_id.macro_file().is_some() {
        // FIXME: Our infra can't handle allow from within macro expansions rn
        return None;
    }
    let diagnostic_range = ctx.sema.diagnostics_display_range(ast);
    // The range for the Actual Name. We don't want to replace the entire declaration. Using the diagnostic range causes issues within in Array Destructuring.
    let name_range = d
        .local
        .primary_source(ctx.sema.db)
        .name()
        .map(|v| v.syntax().original_file_range_rooted(ctx.sema.db))
        .filter(|it| {
            Some(it.file_id) == ast.file_id.file_id()
                && diagnostic_range.range.contains_range(it.range)
        });
    let var_name = d.local.name(ctx.sema.db);
    Some(
        Diagnostic::new_with_syntax_node_ptr(
            ctx,
            DiagnosticCode::RustcLint("unused_variables"),
            "unused variable",
            ast,
        )
        .with_fixes(name_range.and_then(|it| {
            fixes(
                ctx.sema.db,
                var_name,
                it.range,
                diagnostic_range,
                ast.file_id.is_macro(),
                ctx.edition,
            )
        })),
    )
}

fn fixes(
    db: &RootDatabase,
    var_name: Name,
    name_range: TextRange,
    diagnostic_range: FileRange,
    is_in_marco: bool,
    edition: Edition,
) -> Option<Vec<Assist>> {
    if is_in_marco {
        return None;
    }

    Some(vec![Assist {
        id: AssistId::quick_fix("unscore_unused_variable_name"),
        label: Label::new(format!(
            "Rename unused {} to _{}",
            var_name.display(db, edition),
            var_name.display(db, edition)
        )),
        group: None,
        target: diagnostic_range.range,
        source_change: Some(SourceChange::from_text_edit(
            diagnostic_range.file_id,
            TextEdit::replace(name_range, format!("_{}", var_name.display(db, edition))),
        )),
        command: None,
    }])
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_diagnostics, check_fix};

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
        check_diagnostics(
            r#"
macro_rules! my_macro {
    () => {
        let x = 3;
    };
}

fn main() {
    my_macro!();
}
"#,
        );
    }
    #[test]
    fn unused_variable_in_array_destructure() {
        check_fix(
            r#"
fn main() {
    let arr = [1, 2, 3, 4, 5];
    let [_x, y$0 @ ..] = arr;
}
"#,
            r#"
fn main() {
    let arr = [1, 2, 3, 4, 5];
    let [_x, _y @ ..] = arr;
}
"#,
        );
    }

    // regression test as we used to panic in this scenario
    #[test]
    fn unknown_struct_pattern_param_type() {
        check_diagnostics(
            r#"
struct S { field : u32 }
fn f(S { field }: error) {
      // ^^^^^ 💡 warn: unused variable
}
"#,
        );
    }
}
