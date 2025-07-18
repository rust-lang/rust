use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext, Severity};

// Diagnostic: macro-error
//
// This diagnostic is shown for macro expansion errors.

// Diagnostic: attribute-expansion-disabled
//
// This diagnostic is shown for attribute proc macros when attribute expansions have been disabled.

// Diagnostic: proc-macro-disabled
//
// This diagnostic is shown for proc macros that have been specifically disabled via `rust-analyzer.procMacro.ignored`.
pub(crate) fn macro_error(ctx: &DiagnosticsContext<'_>, d: &hir::MacroError) -> Diagnostic {
    // Use more accurate position if available.
    let display_range = ctx.resolve_precise_location(&d.node, d.precise_location);
    Diagnostic::new(
        DiagnosticCode::Ra(d.kind, if d.error { Severity::Error } else { Severity::WeakWarning }),
        d.message.clone(),
        display_range,
    )
    .stable()
}

// Diagnostic: macro-def-error
//
// This diagnostic is shown for macro expansion errors.
pub(crate) fn macro_def_error(ctx: &DiagnosticsContext<'_>, d: &hir::MacroDefError) -> Diagnostic {
    // Use more accurate position if available.
    let display_range =
        ctx.resolve_precise_location(&d.node.map(|it| it.syntax_node_ptr()), d.name);
    Diagnostic::new(
        DiagnosticCode::Ra("macro-def-error", Severity::Error),
        d.message.clone(),
        display_range,
    )
    .stable()
}

#[cfg(test)]
mod tests {
    use crate::{
        DiagnosticsConfig,
        tests::{check_diagnostics, check_diagnostics_with_config},
    };

    #[test]
    fn builtin_macro_fails_expansion() {
        check_diagnostics(
            r#"
#[rustc_builtin_macro]
macro_rules! include { () => {} }

#[rustc_builtin_macro]
macro_rules! compile_error { () => {} }

  include!("doesntexist");
         //^^^^^^^^^^^^^ error: failed to load file `doesntexist`

  compile_error!("compile_error macro works");
//^^^^^^^^^^^^^ error: compile_error macro works

  compile_error! { "compile_error macro braced works" }
//^^^^^^^^^^^^^ error: compile_error macro braced works
            "#,
        );
    }

    #[test]
    fn eager_macro_concat() {
        check_diagnostics(
            r#"
//- /lib.rs crate:lib deps:core
use core::{panic, concat};

mod private {
    pub use core::concat;
}

macro_rules! m {
    () => {
        panic!(concat!($crate::private::concat!("")));
    };
}

fn f() {
    m!();
}

//- /core.rs crate:core
#[macro_export]
#[rustc_builtin_macro]
macro_rules! concat { () => {} }

pub macro panic {
    ($msg:expr) => (
        $crate::panicking::panic_str($msg)
    ),
}
            "#,
        );
    }

    #[test]
    fn include_macro_should_allow_empty_content() {
        let mut config = DiagnosticsConfig::test_sample();

        // FIXME: This is a false-positive, the file is actually linked in via
        // `include!` macro
        config.disabled.insert("unlinked-file".to_owned());

        check_diagnostics_with_config(
            config,
            r#"
//- /lib.rs
#[rustc_builtin_macro]
macro_rules! include { () => {} }

include!("foo/bar.rs");
//- /foo/bar.rs
// empty
"#,
        );
    }

    #[test]
    fn good_out_dir_diagnostic() {
        // FIXME: The diagnostic here is duplicated for each eager expansion
        check_diagnostics(
            r#"
#[rustc_builtin_macro]
macro_rules! include { () => {} }
#[rustc_builtin_macro]
macro_rules! env { () => {} }
#[rustc_builtin_macro]
macro_rules! concat { () => {} }

  include!(concat!(env!("OUT_DIR"), "/out.rs"));
                      //^^^^^^^^^ error: `OUT_DIR` not set, build scripts may have failed to run
                 //^^^^^^^^^^^^^^^ error: `OUT_DIR` not set, build scripts may have failed to run
         //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ error: `OUT_DIR` not set, build scripts may have failed to run
"#,
        );
    }

    #[test]
    fn register_attr_and_tool() {
        cov_mark::check!(register_attr);
        cov_mark::check!(register_tool);
        check_diagnostics(
            r#"
#![register_tool(tool)]
#![register_attr(attr)]

#[tool::path]
#[attr]
struct S;
"#,
        );
        // NB: we don't currently emit diagnostics here
    }

    #[test]
    fn macro_diag_builtin() {
        check_diagnostics(
            r#"
//- minicore: fmt
#[rustc_builtin_macro]
macro_rules! env {}

#[rustc_builtin_macro]
macro_rules! include {}

#[rustc_builtin_macro]
macro_rules! compile_error {}
#[rustc_builtin_macro]
macro_rules! concat {}

fn main() {
    // Test a handful of built-in (eager) macros:

    include!(invalid);
           //^^^^^^^ error: expected string literal
    include!("does not exist");
           //^^^^^^^^^^^^^^^^ error: failed to load file `does not exist`

    include!(concat!("does ", "not ", "exist"));
                  //^^^^^^^^^^^^^^^^^^^^^^^^^^ error: failed to load file `does not exist`

    env!(invalid);
       //^^^^^^^ error: expected string literal

    env!("OUT_DIR");
       //^^^^^^^^^ error: `OUT_DIR` not set, build scripts may have failed to run

    compile_error!("compile_error works");
  //^^^^^^^^^^^^^ error: compile_error works

    // Lazy:

    format_args!();
  //^^^^^^^^^^^ error: Syntax Error in Expansion: expected expression
}
"#,
        );
    }

    #[test]
    fn macro_rules_diag() {
        check_diagnostics(
            r#"
macro_rules! m {
    () => {};
}
fn f() {
    m!();

    m!(hi);
    //^ error: leftover tokens
}
      "#,
        );
    }

    #[test]
    fn dollar_crate_in_builtin_macro() {
        check_diagnostics(
            r#"
#[macro_export]
#[rustc_builtin_macro]
macro_rules! format_args {}

#[macro_export]
macro_rules! arg { () => {} }

#[macro_export]
macro_rules! outer {
    () => {
        $crate::format_args!( "", $crate::arg!(1) )
    };
}

fn f() {
    outer!();
} //^^^^^^ error: leftover tokens
  //^^^^^^ error: Syntax Error in Expansion: expected expression
"#,
        )
    }

    #[test]
    fn def_diagnostic() {
        check_diagnostics(
            r#"
macro_rules! foo {
           //^^^ error: expected subtree
    f => {};
}

fn f() {
    foo!();
  //^^^ error: macro definition has parse errors

}
"#,
        )
    }

    #[test]
    fn expansion_syntax_diagnostic() {
        check_diagnostics(
            r#"
macro_rules! foo {
    () => { struct; };
}

fn f() {
    foo!();
  //^^^ error: Syntax Error in Expansion: expected a name
}
"#,
        )
    }

    #[test]
    fn include_does_not_break_diagnostics() {
        check_diagnostics(
            r#"
//- minicore: include
//- /lib.rs crate:lib
include!("include-me.rs");
//- /include-me.rs
/// long doc that pushes the diagnostic range beyond the first file's text length
  #[err]
//^^^^^^error: unresolved macro `err`
mod prim_never {}
"#,
        );
    }

    #[test]
    fn no_stack_overflow_for_missing_binding() {
        check_diagnostics(
            r#"
#[macro_export]
macro_rules! boom {
    (
        $($code:literal),+,
        $(param: $param:expr,)?
    ) => {{
        let _ = $crate::boom!(@param $($param)*);
    }};
    (@param) => { () };
    (@param $param:expr) => { $param };
}

fn it_works() {
    // NOTE: there is an error, but RA crashes before showing it
    boom!("RAND", param: c7.clone());
               // ^^^^^ error: expected literal
}

        "#,
        );
    }
}
