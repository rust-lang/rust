// Verifies that `#[export_visibility = ...]` can override the visibility
// that is normally implied by `#[export_name]` or `#[no_mangle]`.
//
// High-level test expectations for items with `#[export_name = ...]`
// (or with `#[no_mangle]`) and:
//
// * Without `#[export_visibility = ...]` => public
// * `#[export_visibility = "target_default"]` => value inherited from the target
//   platform or from the `-Zdefault-visibility=...` command-line flag
//   (this expectation depends on whether the `...-HIDDEN` vs `...-PROTECTED`
//   test revisions are used).
//
// Note that what we call "public" in the expectations above is also referred
// to as "default" in LLVM docs - see
// https://llvm.org/docs/LangRef.html#visibility-styles

//@ revisions: LINUX-X86-HIDDEN LINUX-X86-PROTECTED
//@[LINUX-X86-HIDDEN] compile-flags: -Zdefault-visibility=hidden
//@[LINUX-X86-PROTECTED] compile-flags: -Zdefault-visibility=protected

// Exact LLVM IR differs depending on the target triple (e.g. `hidden constant`
// vs `internal constant` vs `constant`).  Because of this, we only apply the
// specific test expectations below to one specific target triple.
//
// Note that `tests/run-make/cdylib-export-visibility` provides similar
// test coverage, but in an LLVM-IR-agnostic / platform-agnostic way.
//@[LINUX-X86-HIDDEN] needs-llvm-components: x86
//@[LINUX-X86-HIDDEN] compile-flags: --target x86_64-unknown-linux-gnu
//@[LINUX-X86-PROTECTED] needs-llvm-components: x86
//@[LINUX-X86-PROTECTED] compile-flags: --target x86_64-unknown-linux-gnu

// This test focuses on rlib to exercise the scenario described in
// https://github.com/rust-lang/rust/issues/73958#issuecomment-2891711649
#![crate_type = "rlib"]
#![feature(export_visibility)]
// Relying on `minicore` makes it easier to run the test, even if the host is
// not a linux-x86 machine.
//@ add-minicore
//@ edition: 2024
#![feature(no_core)]
#![no_core]
use minicore::*;

///////////////////////////////////////////////////////////////////////
// The tests below focus on how `#[export_visibility = ...]` works for
// a `static`.  The tests are based on similar tests in
// `tests/codegen/default-visibility.rs`

#[unsafe(export_name = "static_export_name_no_attr")]
pub static TEST_STATIC_NO_ATTR: u32 = 1101;

#[unsafe(export_name = "static_export_name_target_default")]
#[export_visibility = "target_default"]
pub static TESTED_STATIC_ATTR_ASKS_TO_TARGET_DEFAULT: u32 = 1102;

#[unsafe(no_mangle)]
pub static static_no_mangle_no_attr: u32 = 1201;

#[unsafe(no_mangle)]
#[export_visibility = "target_default"]
pub static static_no_mangle_target_default: u32 = 1202;

// LINUX-X86-HIDDEN: @static_export_name_no_attr = local_unnamed_addr constant
// LINUX-X86-HIDDEN: @static_export_name_target_default = hidden local_unnamed_addr constant
// LINUX-X86-HIDDEN: @static_no_mangle_no_attr = local_unnamed_addr constant
// LINUX-X86-HIDDEN: @static_no_mangle_target_default = hidden local_unnamed_addr constant

// LINUX-X86-PROTECTED: @static_export_name_no_attr = local_unnamed_addr constant
// LINUX-X86-PROTECTED: @static_export_name_target_default = protected local_unnamed_addr constant
// LINUX-X86-PROTECTED: @static_no_mangle_no_attr = local_unnamed_addr constant
// LINUX-X86-PROTECTED: @static_no_mangle_target_default = protected local_unnamed_addr constant

///////////////////////////////////////////////////////////////////////
// The tests below focus on how `#[export_visibility = ...]` works for
// a `fn`.
//
// The tests below try to mimics how `cxx` exports known/hardcoded helpers (e.g.
// `cxxbridge1$string$drop` [1]) as well as build-time-generated thunks (e.g.
// `serde_json_lenient$cxxbridge1$decode_json` from https://crbug.com/418073233#comment7).
//
// [1]
// https://github.com/dtolnay/cxx/blob/ebdd6a0c63ae10dc5224ed21970b7a0504657434/src/symbols/rust_string.rs#L83-L86

#[unsafe(export_name = "test_fn_no_attr")]
unsafe extern "C" fn test_fn_no_attr() -> u32 {
    // We return a unique integer to ensure that each function has a unique body
    // and therefore that identical code folding (ICF) won't fold the functions
    // when linking.
    2001
}

#[unsafe(export_name = "test_fn_target_default")]
#[export_visibility = "target_default"]
unsafe extern "C" fn test_fn_asks_for_target_default() -> u32 {
    2002
}

// LINUX-X86-HIDDEN: define noundef i32 @test_fn_no_attr
// LINUX-X86-HIDDEN: define hidden noundef i32 @test_fn_target_default

// LINUX-X86-PROTECTED: define noundef i32 @test_fn_no_attr
// LINUX-X86-PROTECTED: define protected noundef i32 @test_fn_target_default
