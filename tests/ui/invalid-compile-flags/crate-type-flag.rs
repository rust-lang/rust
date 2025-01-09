//! Check that `rustc`'s `--crate-type` flag accepts `--crate-type=<valid_type>` as well as the
//! multi-value version `--crate-type=<valid_type_1>,<valid_type_2>`.
//!
//! This test does not try to check if the output artifacts are valid.

// FIXME(#132309): add a proper `supports-crate-type` directive.

// Single valid crate types should pass
//@ revisions: lib rlib staticlib dylib cdylib bin proc_dash_macro

//@[lib] compile-flags: --crate-type=lib
//@[lib] check-pass

//@[rlib] compile-flags: --crate-type=rlib
//@[rlib] check-pass

//@[staticlib] compile-flags: --crate-type=staticlib
//@[staticlib] check-pass

//@[dylib] ignore-musl (dylibs are not supported)
//@[dylib] ignore-wasm (dylibs are not supported)
//@[dylib] compile-flags: --crate-type=dylib
//@[dylib] check-pass

//@[cdylib] ignore-musl (cdylibs are not supported)
//@[cdylib] compile-flags: --crate-type=cdylib
//@[cdylib] check-pass

//@[bin] compile-flags: --crate-type=bin
//@[bin] check-pass

//@[proc_dash_macro] ignore-wasm (proc-macro is not supported)
//@[proc_dash_macro] needs-unwind (panic=abort causes warning to be emitted)
//@[proc_dash_macro] compile-flags: --crate-type=proc-macro
//@[proc_dash_macro] check-pass

//@ revisions: multivalue multivalue_combined

//@[multivalue] compile-flags: --crate-type=lib,rlib,staticlib
//@[multivalue] check-pass

//@[multivalue_combined] ignore-musl (dylibs are not supported)
//@[multivalue_combined] ignore-wasm (dylibs are not supported)
//@[multivalue_combined] compile-flags: --crate-type=lib,rlib,staticlib --crate-type=dylib
//@[multivalue_combined] check-pass

// `proc-macro` is accepted, but `proc_macro` is not.
//@ revisions: proc_underscore_macro
//@[proc_underscore_macro] compile-flags: --crate-type=proc_macro
//@[proc_underscore_macro] error-pattern: "unknown crate type: `proc_macro`"

// Empty `--crate-type` not accepted.
//@ revisions: empty_crate_type
//@[empty_crate_type] compile-flags: --crate-type=
//@[empty_crate_type] error-pattern: "unknown crate type: ``"

// Random unknown crate type. Also check that we can handle non-ASCII.
//@ revisions: unknown
//@[unknown] compile-flags: --crate-type=🤡
//@[unknown] error-pattern: "unknown crate type: `🤡`"

fn main() {}
