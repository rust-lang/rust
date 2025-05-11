// This test exercises `-Zremap-path-scope`, macros (like file!()) and dependency.
//
// We test different combinations with/without remap in deps, with/without remap in
// this crate but always in deps and always here but never in deps.

//@ run-pass
//@ check-run-results

//@ revisions: with-diag-in-deps with-macro-in-deps with-debuginfo-in-deps
//@ revisions: only-diag-in-deps only-macro-in-deps only-debuginfo-in-deps
//@ revisions: not-macro-in-deps

//@[with-diag-in-deps] compile-flags: --remap-path-prefix={{src-base}}=remapped
//@[with-macro-in-deps] compile-flags: --remap-path-prefix={{src-base}}=remapped
//@[with-debuginfo-in-deps] compile-flags: --remap-path-prefix={{src-base}}=remapped
//@[not-macro-in-deps] compile-flags: --remap-path-prefix={{src-base}}=remapped

//@[with-diag-in-deps] compile-flags: -Zremap-path-scope=diagnostics
//@[with-macro-in-deps] compile-flags: -Zremap-path-scope=macro
//@[with-debuginfo-in-deps] compile-flags: -Zremap-path-scope=debuginfo
//@[not-macro-in-deps] compile-flags: -Zremap-path-scope=macro

//@[with-diag-in-deps] aux-build:file-diag.rs
//@[with-macro-in-deps] aux-build:file-macro.rs
//@[with-debuginfo-in-deps] aux-build:file-debuginfo.rs
//@[only-diag-in-deps] aux-build:file-diag.rs
//@[only-macro-in-deps] aux-build:file-macro.rs
//@[only-debuginfo-in-deps] aux-build:file-debuginfo.rs
//@[not-macro-in-deps] aux-build:file.rs

#[cfg(any(with_diag_in_deps, only_diag_in_deps))]
extern crate file_diag as file;

#[cfg(any(with_macro_in_deps, only_macro_in_deps))]
extern crate file_macro as file;

#[cfg(any(with_debuginfo_in_deps, only_debuginfo_in_deps))]
extern crate file_debuginfo as file;

#[cfg(not_macro_in_deps)]
extern crate file;

fn main() {
    println!("file::my_file!() = {}", file::my_file!());
    println!("file::file() = {}", file::file());
    println!("file!() = {}", file!());
}
