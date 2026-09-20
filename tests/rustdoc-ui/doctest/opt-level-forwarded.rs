// See <https://github.com/rust-lang/rust/issues/43031>.

//@ revisions: opt noopt
//@ compile-flags: --test
//@[opt] compile-flags: -O
//@[noopt] compile-flags: -O -Copt-level=0
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@[opt] check-pass
//@[noopt] failure-status: 101

/// ```
/// #[cfg(debug_assertions)]
/// compile_error!("debug assertions are enabled");
/// fn main() {}
/// ```
pub struct Bar;
