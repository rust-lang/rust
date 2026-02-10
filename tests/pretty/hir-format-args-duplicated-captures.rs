//! Test for <https://github.com/rust-lang/rust/issues/145739>: identifiers referring to places
//! should have their captures de-duplicated by `format_args!`, but identifiers referring to values
//! should not be de-duplicated.
//@ pretty-compare-only
//@ pretty-mode:hir
//@ pp-exact:hir-format-args-duplicated-captures.pp

const X: i32 = 42;

struct Struct;

static STATIC: i32 = 0;

fn main() {
    // Consts and constructors refer to values. Whether we construct them multiple times or just
    // once can be observed in some cases, so we don't de-duplicate them.
    let _ = format_args!("{X} {X}");
    let _ = format_args!("{Struct} {Struct}");

    // Variables and statics refer to places. We can de-duplicate without an observable difference.
    let x = 3;
    let _ = format_args!("{STATIC} {STATIC}");
    let _ = format_args!("{x} {x}");

    // We don't de-duplicate widths or precisions since de-duplication can be observed.
    let _ = format_args!("{x:x$} {x:x$}");
    let _ = format_args!("{0:.x$} {0:.x$}", 0.0);
}
