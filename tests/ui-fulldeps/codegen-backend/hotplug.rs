//@ edition: 2021
//@ build-pass
//@ ignore-stage1 (requires matching sysroot built with in-tree compiler)

//@ aux-codegen-backend: the_backend.rs
//@ normalize-stdout: "libthe_backend.dylib" -> "libthe_backend.so"
//@ normalize-stdout: "the_backend.dll" -> "libthe_backend.so"

// Pick a target that requires no target features, so that no warning is shown
// about missing target features.
//@ compile-flags: --target arm-unknown-linux-gnueabi
//@ needs-llvm-components: arm
//@ revisions: normal dep bindep
//@ compile-flags: --crate-type=lib
//@ [normal] compile-flags: --emit=link=-
//@ [dep]    compile-flags: --emit=link,dep-info=-
//@ [bindep] compile-flags: --emit=link,dep-info=- -Zbinary-dep-depinfo

#![feature(no_core)]
#![no_core]

// This test both exists as a check that -Zcodegen-backend is capable of loading external codegen
// backends and that this external codegen backend is only included in the dep info if
// -Zbinary-dep-depinfo is used.
