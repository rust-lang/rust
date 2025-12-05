// Check that only `-C linker-features=-lld` is stable on x64 linux. Any other value or target,
// needs `-Z unstable-options`.

// ignore-tidy-linelength

//@ revisions: unstable_target_positive unstable_target_negative unstable_positive
//@ [unstable_target_negative] compile-flags: --target=x86_64-unknown-linux-musl -C linker-features=-lld --crate-type=rlib
//@ [unstable_target_negative] needs-llvm-components: x86
//@ [unstable_target_positive] compile-flags: --target=x86_64-unknown-linux-musl -C linker-features=+lld --crate-type=rlib
//@ [unstable_target_positive] needs-llvm-components: x86
//@ [unstable_positive] compile-flags: --target=x86_64-unknown-linux-gnu -C linker-features=+lld --crate-type=rlib
//@ [unstable_positive] needs-llvm-components: x86


#![feature(no_core)]
#![no_core]

//[unstable_target_negative]~? ERROR `-C linker-features=-lld` is unstable on the `x86_64-unknown-linux-musl` target
//[unstable_target_positive,unstable_positive]~? ERROR `-C linker-features=+lld` is unstable, and also requires the `-Z unstable-options`
