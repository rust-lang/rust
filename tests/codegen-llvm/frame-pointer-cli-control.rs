//@ add-core-stubs
//@ compile-flags: --crate-type=rlib -Copt-level=0
//@ revisions: force-on aarch64-apple aarch64-apple-on aarch64-apple-off
//@ [force-on] compile-flags: -Cforce-frame-pointers=on
//@ [aarch64-apple] needs-llvm-components: aarch64
//@ [aarch64-apple] compile-flags: --target=aarch64-apple-darwin
//@ [aarch64-apple-on] needs-llvm-components: aarch64
//@ [aarch64-apple-on] compile-flags: --target=aarch64-apple-darwin -Cforce-frame-pointers=on
//@ [aarch64-apple-off] needs-llvm-components: aarch64
//@ [aarch64-apple-off] compile-flags: --target=aarch64-apple-darwin -Cforce-frame-pointers=off
/*!

Tests the extent to which frame pointers can be controlled by the CLI.
The behavior of our frame pointer options, at present, is an irreversible ratchet, where
a "weaker" option that allows omitting frame pointers may be overridden by the target demanding
that all code (or all non-leaf code, more often) must be compiled with frame pointers.
This was discussed on 2025-05-22 in the T-compiler meeting and accepted as an intentional change,
ratifying the prior decisions by compiler contributors and reviewers as correct,
though it was also acknowledged that the flag allows somewhat confusing inputs.

We find aarch64-apple-darwin useful because of its icy-clear policy regarding frame pointers,
e.g. <https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms> says:

* The frame pointer register (x29) must always address a valid frame record. Some functions —
  such as leaf functions or tail calls — may opt not to create an entry in this list.
  As a result, stack traces are always meaningful, even without debug information.

Many Rust fn, if externally visible, may be expected to follow target ABI by tools or asm code!
This can make it a problem to generate ABI-incorrect code, which may mean "with frame pointers".
For this and other reasons, `-Cforce-frame-pointers=off` cannot override the target definition.
This can cause some confusion because it is "reverse polarity" relative to C compilers, which have
commands like `-fomit-frame-pointer`, `-fomit-leaf-frame-pointer`, or `-fno-omit-frame-pointer`!

Specific cases where platforms or tools rely on frame pointers for sound or correct unwinding:
- illumos: <https://smartos.org/bugview/OS-7515>
- aarch64-windows: <https://github.com/rust-lang/rust/issues/123686>
- aarch64-linux: <https://github.com/rust-lang/rust/issues/123733>
- dtrace (freebsd and openbsd): <https://github.com/rust-lang/rust/issues/97723>
- openbsd: <https://github.com/rust-lang/rust/issues/43575>
- i686-msvc <https://github.com/rust-lang/backtrace-rs/pull/584#issuecomment-1966177530>
- i686-mingw: <https://github.com/rust-lang/rust/commit/3f1d3948d6d434b34dd47f132c126a6cb6b8a4ab>
*/
#![feature(no_core, lang_items)]
#![no_core]

extern crate minicore;

// CHECK: i32 @peach{{.*}}[[PEACH_ATTRS:\#[0-9]+]] {
#[no_mangle]
pub fn peach(x: u32) -> u32 {
    x
}

// CHECK: attributes [[PEACH_ATTRS]] = {
// force-on-SAME: {{.*}}"frame-pointer"="all"
// aarch64-apple-SAME: {{.*}}"frame-pointer"="non-leaf"
// aarch64-apple-on-SAME: {{.*}}"frame-pointer"="all"
//
// yes, we are testing this doesn't do anything:
// aarch64-apple-off-SAME: {{.*}}"frame-pointer"="non-leaf"
// CHECK-SAME: }
