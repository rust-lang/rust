use hir::Name;
use ide_db::text_edit::TextEdit;
use ide_db::{
    FileRange, RootDatabase,
    assists::{Assist, AssistId},
    label::Label,
    source_change::SourceChange,
};
use syntax::{AstNode, Edition, TextRange, ToSmolStr};

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
    let primary_source = d.local.primary_source(ctx.sema.db);
    let name_range = primary_source
        .name()
        .map(|v| v.syntax().original_file_range_rooted(ctx.sema.db))
        .filter(|it| {
            Some(it.file_id) == ast.file_id.file_id()
                && diagnostic_range.range.contains_range(it.range)
        });
    let is_shorthand_field = primary_source
        .source
        .value
        .left()
        .and_then(|name| name.syntax().parent())
        .and_then(syntax::ast::RecordPatField::cast)
        .is_some_and(|field| field.colon_token().is_none());
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
                is_shorthand_field,
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
    is_shorthand_field: bool,
    edition: Edition,
) -> Option<Vec<Assist>> {
    if is_in_marco {
        return None;
    }
    let name = var_name.display(db, edition).to_smolstr();
    let name = name.strip_prefix("r#").unwrap_or(&name);
    let new_name = if is_shorthand_field { format!("{name}: _{name}") } else { format!("_{name}") };

    Some(vec![Assist {
        id: AssistId::quick_fix("unscore_unused_variable_name"),
        label: Label::new(format!("Rename unused {name} to {new_name}")),
        group: None,
        target: diagnostic_range.range,
        source_change: Some(SourceChange::from_text_edit(
            diagnostic_range.file_id,
            TextEdit::replace(name_range, new_name),
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
    fn apply_last_lint_attribute_when_multiple_are_present() {
        check_diagnostics(
            r#"
#![allow(unused_variables)]
#![warn(unused_variables)]
#![deny(unused_variables)]

fn main() {
    let x = 2;
      //^ 💡 error: unused variable

    #[deny(unused_variables)]
    #[warn(unused_variables)]
    #[allow(unused_variables)]
    let y = 0;
}
"#,
        );
    }

    #[test]
    fn prefer_closest_ancestor_lint_attribute() {
        check_diagnostics(
            r#"
#![allow(unused_variables)]

fn main() {
    #![warn(unused_variables)]

    #[deny(unused_variables)]
    let x = 2;
      //^ 💡 error: unused variable
}

#[warn(unused_variables)]
fn main2() {
    #[deny(unused_variables)]
    let x = 2;
      //^ 💡 error: unused variable
}

#[warn(unused_variables)]
fn main3() {
    let x = 2;
      //^ 💡 warn: unused variable
}

fn main4() {
    let x = 2;
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
        Foo { f1: _f1, f2 } => {
            _ = f2;
        }
    }
}
"#,
        );

        check_fix(
            r#"
fn main() {
    let $0r#type = 2;
}
"#,
            r#"
fn main() {
    let _type = 2;
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

    #[test]
    fn unused_variable_in_record_field() {
        check_fix(
            r#"
struct S { field : u32 }
fn main() {
    let s = S { field : 2 };
    let S { field: $0x } = s
}
"#,
            r#"
struct S { field : u32 }
fn main() {
    let s = S { field : 2 };
    let S { field: _x } = s
}
"#,
        );
    }

    #[test]
    fn unused_variable_in_shorthand_record_field() {
        check_fix(
            r#"
struct S { field : u32 }
fn main() {
    let s = S { field : 2 };
    let S { $0field } = s
}
"#,
            r#"
struct S { field : u32 }
fn main() {
    let s = S { field : 2 };
    let S { field: _field } = s
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

    #[test]
    fn crate_attrs_lint_smoke_test() {
        check_diagnostics(
            r#"
//- /lib.rs crate:foo crate-attr:deny(unused_variables)
fn main() {
    let x = 2;
      //^ 💡 error: unused variable
}
"#,
        );
    }

    #[test]
    fn crate_attrs_should_not_override_lints_in_source() {
        check_diagnostics(
            r#"
//- /lib.rs crate:foo crate-attr:allow(unused_variables)
#![deny(unused_variables)]
fn main() {
    let x = 2;
      //^ 💡 error: unused variable
}
"#,
        );
    }

    #[test]
    fn crate_attrs_should_preserve_lint_order() {
        check_diagnostics(
            r#"
//- /lib.rs crate:foo crate-attr:allow(unused_variables) crate-attr:warn(unused_variables)
fn main() {
    let x = 2;
      //^ 💡 warn: unused variable
}
"#,
        );
    }
}
