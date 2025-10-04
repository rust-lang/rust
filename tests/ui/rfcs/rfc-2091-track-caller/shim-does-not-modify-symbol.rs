//@ run-pass
#![feature(rustc_attrs)]

// The shim that is generated for a function annotated with `#[track_caller]` should not inherit
// attributes that modify its symbol name. Failing to remove these attributes from the shim
// leads to errors like `symbol `foo` is already defined`.
//
// See also https://github.com/rust-lang/rust/issues/143162.

#[unsafe(no_mangle)]
#[track_caller]
pub fn foo() {}

#[unsafe(export_name = "bar")]
#[track_caller]
pub fn bar() {}

#[rustc_std_internal_symbol]
#[track_caller]
pub fn baz() {}

fn main() {
    let _a = foo as fn();
    let _b = bar as fn();
    let _c = baz as fn();
}
