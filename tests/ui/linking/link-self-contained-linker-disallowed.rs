// ignore-tidy-file-linelength
//
// Check that only `-C link-self-contained=-linker` is stable on x64 and loongarch64 linux.
// Any other value or target, needs `-Z unstable-options`.

//@ revisions: unstable_target_positive_x86_64 unstable_target_positive_loongarch64 unstable_target_negative_x86_64 unstable_target_negative_loongarch64 unstable_positive_x86_64 unstable_positive_loongarch64
//@ [unstable_target_negative_x86_64] compile-flags: --target=x86_64-unknown-linux-musl -C link-self-contained=-linker --crate-type=rlib
//@ [unstable_target_negative_x86_64] needs-llvm-components: x86
//@ [unstable_target_negative_loongarch64] compile-flags: --target=loongarch64-unknown-linux-musl -C link-self-contained=-linker --crate-type=rlib
//@ [unstable_target_negative_loongarch64] needs-llvm-components: loongarch
//@ [unstable_target_positive_x86_64] compile-flags: --target=x86_64-unknown-linux-musl -C link-self-contained=+linker --crate-type=rlib
//@ [unstable_target_positive_x86_64] needs-llvm-components: x86
//@ [unstable_target_positive_loongarch64] compile-flags: --target=loongarch64-unknown-linux-musl -C link-self-contained=+linker --crate-type=rlib
//@ [unstable_target_positive_loongarch64] needs-llvm-components: loongarch
//@ [unstable_positive_x86_64] compile-flags: --target=x86_64-unknown-linux-gnu -C link-self-contained=+linker --crate-type=rlib
//@ [unstable_positive_x86_64] needs-llvm-components: x86
//@ [unstable_positive_loongarch64] compile-flags: --target=loongarch64-unknown-linux-gnu -C link-self-contained=+linker --crate-type=rlib
//@ [unstable_positive_loongarch64] needs-llvm-components: loongarch

#![feature(no_core)]
#![no_core]

//[unstable_target_negative_x86_64]~? ERROR `-C link-self-contained=-linker` is unstable on the `x86_64-unknown-linux-musl` target
//[unstable_target_negative_loongarch64]~? ERROR `-C link-self-contained=-linker` is unstable on the `loongarch64-unknown-linux-musl` target
//[unstable_target_positive_x86_64,unstable_target_positive_loongarch64,unstable_positive_x86_64,unstable_positive_loongarch64]~? ERROR only `-C link-self-contained` values `y`/`yes`/`on`/`n`/`no`/`off`/`-linker` are stable
