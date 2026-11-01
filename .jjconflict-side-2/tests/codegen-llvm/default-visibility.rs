// Verifies that `Session::default_visibility` is affected when using the related cmdline
// flag.  This is a regression test for https://github.com/rust-lang/compiler-team/issues/782.  See
// also https://github.com/rust-lang/rust/issues/73295 and
// https://github.com/rust-lang/rust/issues/37530.

//@ revisions:DEFAULT HIDDEN PROTECTED INTERPOSABLE
//@[HIDDEN] compile-flags: -Zdefault-visibility=hidden
//@[PROTECTED] compile-flags: -Zdefault-visibility=protected
//@[INTERPOSABLE] compile-flags: -Zdefault-visibility=interposable

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

// HIDDEN:       @{{.*}}default_visibility{{.*}}tested_symbol{{.*}} = hidden constant
// PROTECTED:    @{{.*}}default_visibility{{.*}}tested_symbol{{.*}} = protected constant
// INTERPOSABLE: @{{.*}}default_visibility{{.*}}tested_symbol{{.*}} = constant
// DEFAULT:      @{{.*}}default_visibility{{.*}}tested_symbol{{.*}} = constant

#[inline(never)]
pub fn do_memcmp(left: &[u8], right: &[u8]) -> i32 {
    left.cmp(right) as i32
}

// CHECK: define {{.*}} @{{.*}}do_memcmp{{.*}} {
// CHECK: }

// `do_memcmp` should invoke core::intrinsic::compare_bytes which emits a call
// to the C symbol `memcmp` (at least on x86_64-unknown-linux-gnu). This symbol
// should *not* be declared hidden or protected.

// HIDDEN:       declare i32 @memcmp
// PROTECTED:    declare i32 @memcmp
// INTERPOSABLE: declare i32 @memcmp
// DEFAULT:      declare i32 @memcmp
