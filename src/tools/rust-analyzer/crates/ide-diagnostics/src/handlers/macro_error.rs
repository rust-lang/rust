use crate::{Diagnostic, DiagnosticsContext};

// Diagnostic: macro-error
//
// This diagnostic is shown for macro expansion errors.
pub(crate) fn macro_error(ctx: &DiagnosticsContext<'_>, d: &hir::MacroError) -> Diagnostic {
    // Use more accurate position if available.
    let display_range = ctx.resolve_precise_location(&d.node, d.precise_location);
    Diagnostic::new("macro-error", d.message.clone(), display_range).experimental()
}

// Diagnostic: macro-error
//
// This diagnostic is shown for macro expansion errors.
pub(crate) fn macro_def_error(ctx: &DiagnosticsContext<'_>, d: &hir::MacroDefError) -> Diagnostic {
    // Use more accurate position if available.
    let display_range =
        ctx.resolve_precise_location(&d.node.clone().map(|it| it.syntax_node_ptr()), d.name);
    Diagnostic::new("macro-def-error", d.message.clone(), display_range).experimental()
}

#[cfg(test)]
mod tests {
    use crate::{
        tests::{check_diagnostics, check_diagnostics_with_config},
        DiagnosticsConfig,
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
//^^^^^^^ error: failed to load file `doesntexist`

  compile_error!("compile_error macro works");
//^^^^^^^^^^^^^ error: compile_error macro works
            "#,
        );
    }

    #[test]
    fn eager_macro_concat() {
        // FIXME: this is incorrectly handling `$crate`, resulting in a wrong diagnostic.
        // See: https://github.com/rust-lang/rust-analyzer/issues/10300

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
  //^^^^ error: unresolved macro `$crate::private::concat!`
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
        config.disabled.insert("unlinked-file".to_string());

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
        check_diagnostics(
            r#"
#[rustc_builtin_macro]
macro_rules! include { () => {} }
#[rustc_builtin_macro]
macro_rules! env { () => {} }
#[rustc_builtin_macro]
macro_rules! concat { () => {} }

  include!(concat!(env!("OUT_DIR"), "/out.rs"));
//^^^^^^^ error: `OUT_DIR` not set, enable "build scripts" to fix
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
#[rustc_builtin_macro]
macro_rules! env {}

#[rustc_builtin_macro]
macro_rules! include {}

#[rustc_builtin_macro]
macro_rules! compile_error {}

#[rustc_builtin_macro]
macro_rules! format_args { () => {} }

fn main() {
    // Test a handful of built-in (eager) macros:

    include!(invalid);
  //^^^^^^^ error: could not convert tokens
    include!("does not exist");
  //^^^^^^^ error: failed to load file `does not exist`

    env!(invalid);
  //^^^ error: could not convert tokens

    env!("OUT_DIR");
  //^^^ error: `OUT_DIR` not set, enable "build scripts" to fix

    compile_error!("compile_error works");
  //^^^^^^^^^^^^^ error: compile_error works

    // Lazy:

    format_args!();
  //^^^^^^^^^^^ error: no rule matches input tokens
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
} //^^^^^^^^ error: leftover tokens
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
  //^^^ error: invalid macro definition: expected subtree

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
}
