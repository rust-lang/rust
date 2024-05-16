//@ compile-flags: --crate-type=rlib -Copt-level=0
//@ revisions: force-on aarch64-apple aarch64-apple-on aarch64-apple-off
//@ [force-on] compile-flags: -Cforce-frame-pointers=on
//@ [aarch64-apple] needs-llvm-components: aarch64
//@ [aarch64-apple] compile-flags: --target=aarch64-apple-darwin
//@ [aarch64-apple-on] needs-llvm-components: aarch64
//@ [aarch64-apple-on] compile-flags: --target=aarch64-apple-darwin -Cforce-frame-pointers=on
//@ [aarch64-apple-off] needs-llvm-components: aarch64
//@ [aarch64-apple-off] compile-flags: --target=aarch64-apple-darwin -Cforce-frame-pointers=off
/*
Tests that the frame pointers can be controlled by the CLI. We find aarch64-apple-darwin useful
because of its icy-clear policy regarding frame pointers (software SHALL be compiled with them),
e.g. <https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms> says:

* The frame pointer register (x29) must always address a valid frame record. Some functions —
  such as leaf functions or tail calls — may opt not to create an entry in this list.
  As a result, stack traces are always meaningful, even without debug information.
*/
// Want the links to be clickable? Try:
// rustdoc +nightly ./tests/codegen/frame-pointer-cli-control.rs --out-dir=dilemma && xdg-open ./dilemma/frame_pointer_cli_control/index.htm
//!
//! TODO: T-compiler needs to make a decision about whether or not rustc should defer to the CLI
//! when given the -Cforce-frame-pointers flag, even when the target platform mandates it for ABI!
//! Considerations:
//!
//! - Many Rust fn, if externally visible, may be expected to follow ABI by tools or even asm code!
//!   This can potentially make it unsound to generate ABI-incorrect (without frame pointers) code.
//! - Some platforms (e.g. illumos) seem to have unwinding completely break without frame-pointers:
//!   - <https://smartos.org/bugview/OS-7515>
//!   - <https://github.com/rust-lang/rust/blob/b71e8cbaf2c7cae4d36898fff1d0ba19d9233082/compiler/rustc_target/src/spec/base/illumos.rs#L35>
//! - The code in more sophisticated backtrace and unwinding routines is notorious "dark magic"
//!   that is poorly documented, understood by few, and prone to regressions for strange reasons.
//!   Such unwinders in-theory allow using uwutables instead of frame pointers, but if they break?
//!   For better or worse, these problems are often mended by forcing frame pointers:
//!   - <https://github.com/rust-lang/rust/issues/123733>
//!   - <https://github.com/rust-lang/rust/issues/104388>
//!   - <https://github.com/rust-lang/rust/issues/97723>
//!   - <https://github.com/rust-lang/rust/issues/43575>
//! - Those 32-bit x86 platforms that reputedly benefit the most from omitting frame pointers?
//!   They also suffer the most from doing so, because of ancient unwinding/backtrace handling:
//!   - <https://github.com/rust-lang/backtrace-rs/pull/624#issuecomment-2109962234>
//!   - <https://github.com/rust-lang/backtrace-rs/pull/584#issuecomment-1952006125>
//!   - <https://github.com/rust-lang/backtrace-rs/pull/601>
//!   - <https://github.com/rust-lang/rust/commit/3f1d3948d6d434b34dd47f132c126a6cb6b8a4ab>
//! - Omitting frame-pointers may not meaningfully impact -Cpanic=abort binaries in function, but
//!   some targets that default to panic=abort do force frame pointers to allow debugging to work,
//!   even when they are notably pressed for space:
//!   - <https://github.com/rust-lang/rust/blob/b71e8cbaf2c7cae4d36898fff1d0ba19d9233082/compiler/rustc_target/src/spec/base/thumb.rs#L51-L53>
//! - There is, of course, the question of whether this compiler option was ever a good idea?
//!   Some would say no: <https://www.brendangregg.com/blog/2024-03-17/the-return-of-the-frame-pointers.html>
//! - Despite all other remarks, it is true that the previous behavior was to defer to the CLI!
//!   It may be considered a regression, but perhaps some platforms rely on that "regression".
//!   Note the current behavior does uphold the doc's "without reading the source" contract:
//!   - <https://github.com/rust-lang/rust/blob/9e7aff794539aa040362f4424eb29207449ffce0/src/doc/rustc/src/codegen-options/index.md?plain=1#L157-L165>

#![feature(no_core, lang_items)]
#![no_core]
#[lang = "sized"]
trait Sized {}
#[lang = "copy"]
trait Copy {}
impl Copy for u32 {}

// CHECK: define i32 @peach{{.*}}[[PEACH_ATTRS:\#[0-9]+]] {
#[no_mangle]
pub fn peach(x: u32) -> u32 {
    x
}

// CHECK: attributes [[PEACH_ATTRS]] = {
// force-on-SAME: {{.*}}"frame-pointer"="all"
// aarch64-apple-SAME: {{.*}}"frame-pointer"="non-leaf"
// aarch64-apple-on-SAME: {{.*}}"frame-pointer"="all"
// aarch64-apple-off-NOT: {{.*}}"frame-pointer"{{.*}}
// CHECK-SAME: }
