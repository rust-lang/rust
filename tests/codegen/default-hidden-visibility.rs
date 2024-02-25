// Verifies that `Session::default_hidden_visibility` is affected when using the related cmdline
// flag.  This is a regression test for https://github.com/rust-lang/compiler-team/issues/656.  See
// also https://github.com/rust-lang/rust/issues/73295 and
// https://github.com/rust-lang/rust/issues/37530.

//@ revisions:DEFAULT YES NO
//@[YES] compile-flags: -Zdefault-hidden-visibility=yes
//@[NO]  compile-flags: -Zdefault-hidden-visibility=no

// The test scenario is specifically about visibility of symbols exported out of dynamically linked
// libraries.
#![crate_type = "dylib"]

// The test scenario needs to use a Rust-public, but non-explicitly-exported symbol
// (e.g. the test doesn't use `#[no_mangle]`, because currently it implies that
// the symbol should be exported;  we don't want that - we want to test the *default*
// export setting instead).
#[used]
pub static tested_symbol: [u8; 6] = *b"foobar";

// Exact LLVM IR differs depending on the target triple (e.g. `hidden constant`
// vs `internal constant` vs `constant`).  Because of this, we only apply the
// specific test expectations below to one specific target triple.  If needed,
// additional targets can be covered by adding copies of this test file with
// a different `only-X` directive.
//
//@     only-x86_64-unknown-linux-gnu

// DEFAULT: @{{.*}}default_hidden_visibility{{.*}}tested_symbol{{.*}} = constant
// YES:     @{{.*}}default_hidden_visibility{{.*}}tested_symbol{{.*}} = hidden constant
// NO:      @{{.*}}default_hidden_visibility{{.*}}tested_symbol{{.*}} = constant
