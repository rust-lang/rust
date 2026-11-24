// Check that only `-C linker-features=-lld` is stable on x64 and loongarch64 linux
// and that other linker features require using `-Z unstable-options`.
//
// Note that, currently, only `lld` is parsed on the CLI, but that other linker features can exist
// internally (`cc`).
//
// ignore-tidy-file-linelength
//
//@ revisions: x86_64 loongarch64
//@ [x86_64] compile-flags: --target=x86_64-unknown-linux-gnu -C linker-features=+cc --crate-type=rlib
//@ [x86_64] needs-llvm-components: x86
//@ [loongarch64] compile-flags: --target=loongarch64-unknown-linux-gnu -C linker-features=+cc --crate-type=rlib
//@ [loongarch64] needs-llvm-components: loongarch

#![feature(no_core)]
#![no_core]

//[x86_64,loongarch64]~? ERROR incorrect value `+cc` for codegen option `linker-features`
