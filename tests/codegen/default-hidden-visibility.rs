// Verifies that `Session::default_hidden_visibility` is affected when using the related cmdline
// flag.  This is a regression test for https://github.com/rust-lang/compiler-team/issues/656.  See
// also https://github.com/rust-lang/rust/issues/73295 and
// https://github.com/rust-lang/rust/issues/37530.

// We test 3 combinations of command-line flags:
// * No extra command-line flag: DEFAULT
// * Overriding to "yes": YES
// * Overriding to "no": NO
//
//     revisions:DEFAULT YES NO
//     [YES] compile-flags: -Zdefault-hidden-visibility=yes
//     [NO]  compile-flags: -Zdefault-hidden-visibility=no

// `compiler/rustc_target/src/spec/base/wasm.rs` has a different default value of
// `default_hidden_visibility` - it wouldn't match the test expectations below.
// And therefore we skip this test on WASM:
//
//     ignore-wasm32

// The test scenario is specifically about visibility of symbols exported out of dynamically linked
// libraries.
#![crate_type = "dylib"]

// The test scenario needs to use a Rust-public, but non-explicitly-exported symbol
// (e.g. the test doesn't use `#[no_mangle]`, because currently it implies that
// the symbol should be exported;  we don't want that - we want to test the *default*
// export setting instead).
// .
// We want to verify that the cmdline flag affects the visibility of this symbol:
//
// DEFAULT: @{{.*}}default_hidden_visibility{{.*}}exported_symbol{{.*}} = constant
// YES:     @{{.*}}default_hidden_visibility{{.*}}exported_symbol{{.*}} = hidden constant
// NO:      @{{.*}}default_hidden_visibility{{.*}}exported_symbol{{.*}} = constant
#[used]
pub static exported_symbol: [u8; 6] = *b"foobar";
