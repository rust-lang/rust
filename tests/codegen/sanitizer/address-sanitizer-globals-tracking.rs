// Verifies that AddressSanitizer symbols show up as expected in LLVM IR
// with -Zsanitizer (DO NOT SUBMIT: add ASAN (no LTO) and ASAN-LTO2 (lto=fat) tests).
//
// Notes about the `compile-flags` below:
//
// * The original issue only reproed with LTO - this is why this angle has
//   extra test coverage via different `revisions`
// * To observe the failure/repro at LLVM-IR level we need to use `staticlib`
//   which necessitates `-C prefer-dynamic=false` - without the latter flag,
//   we would have run into "cannot prefer dynamic linking when performing LTO".
//
// needs-sanitizer-address
//
// revisions:ASAN ASAN-LTO
//[ASAN]     compile-flags: -Zsanitizer=address
//[ASAN-LTO] compile-flags: -Zsanitizer=address -C prefer-dynamic=false -C lto

#![crate_type="staticlib"]

// The test below mimics `CACHED_POW10` from `library/core/src/num/flt2dec/strategy/grisu.rs` which
// (because of incorrect handling of `___asan_globals_registered` during LTO) was incorrectly
// reported as an ODR violation in https://crbug.com/1459233#c1.  Before this bug was fixed,
// `___asan_globals_registered` would show up as `internal global i64`.
//
// See https://github.com/rust-lang/rust/issues/113404 for more discussion.
//
// CHECK: @___asan_globals_registered = common hidden global i64 0
// CHECK: @__start_asan_globals = extern_weak hidden global i64
// CHECK: @__stop_asan_globals = extern_weak hidden global i64
#[no_mangle]
pub static CACHED_POW10: [(u64, i16, i16); 4] = [
    (0xe61acf033d1a45df, -1087, -308),
    (0xab70fe17c79ac6ca, -1060, -300),
    (0xff77b1fcbebcdc4f, -1034, -292),
    (0xbe5691ef416bd60c, -1007, -284),
];
