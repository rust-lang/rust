// Verifies that AddressSanitizer symbols show up as expected in LLVM IR with `-Zsanitizer`.
// This is a regression test for https://github.com/rust-lang/rust/issues/113404
//
// Notes about the `compile-flags` below:
//
// * The original issue only reproed with LTO - this is why this angle has
//   extra test coverage via different `revisions`
// * To observe the failure/repro at LLVM-IR level we need to use `staticlib`
//   which necessitates `-C prefer-dynamic=false` - without the latter flag,
//   we would have run into "cannot prefer dynamic linking when performing LTO".
//
// The test is restricted to `only-linux`, because the sanitizer-related instrumentation is target
// specific.  In particular, `___asan_globals_registered` is only used in the
// `InstrumentGlobalsELF` and `InstrumentGlobalsMachO` code paths.  The `only-linux` filter is
// narrower than really needed (i.e. narrower than ELF-or-MachO), but this seems ok - having a
// linux-only regression test should be sufficient here.
//
//@ needs-sanitizer-address
//@ only-linux
//
//@ revisions:ASAN ASAN-FAT-LTO
//@ compile-flags: -Zsanitizer=address -Ctarget-feature=-crt-static -C unsafe-allow-abi-mismatch=sanitizer
// [ASAN] no extra compile-flags
//@[ASAN-FAT-LTO] compile-flags: -Cprefer-dynamic=false -Clto=fat

#![crate_type = "staticlib"]

// The test below mimics `CACHED_POW10` from `library/core/src/num/flt2dec/strategy/grisu.rs` which
// (because of incorrect handling of `___asan_globals_registered` during LTO) was incorrectly
// reported as an ODR violation in https://crbug.com/1459233#c1.  Before this bug was fixed,
// `___asan_globals_registered` would show up as `internal global i64` rather than `common hidden
// global i64`.  (The test expectations ignore the exact type because on `arm-android` the type
// is `i32` rather than `i64`.)
//
// CHECK: @___asan_globals_registered = common hidden global
// CHECK: @__start_asan_globals = extern_weak hidden global
// CHECK: @__stop_asan_globals = extern_weak hidden global
#[no_mangle]
pub static CACHED_POW10: [(u64, i16, i16); 4] = [
    (0xe61acf033d1a45df, -1087, -308),
    (0xab70fe17c79ac6ca, -1060, -300),
    (0xff77b1fcbebcdc4f, -1034, -292),
    (0xbe5691ef416bd60c, -1007, -284),
];
