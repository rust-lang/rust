//@ ignore-backends: gcc
//@ revisions: empty unprefixed all_unknown all_known mixed

//@ compile-flags: -Zunstable-options
//@[empty] needs-llvm-components: aarch64
//@[empty] compile-flags: --target aarch64-unknown-linux-pauthtest -Tpointer-authentication=
//@[unprefixed] needs-llvm-components: aarch64
//@[unprefixed] compile-flags: --target aarch64-unknown-linux-pauthtest -Tpointer-authentication=auth-traps
//@[all_unknown] needs-llvm-components: aarch64
//@[all_unknown] compile-flags: --target aarch64-unknown-linux-pauthtest -Tpointer-authentication=+I,+do,-not,-exist
//@[all_known] check-pass
//@[all_known] needs-llvm-components: aarch64
//@[all_known] compile-flags: --target aarch64-unknown-linux-pauthtest -Tpointer-authentication=+elf-got,-init-fini
//@[mixed] needs-llvm-components: aarch64
//@[mixed] compile-flags: --target aarch64-unknown-linux-pauthtest -Tpointer-authentication=+elf-got,-imaginary

#![feature(no_core)]
#![no_std]
#![no_main]
#![no_core]

//[empty]~? ERROR incorrect value `` for codegen option `pointer-authentication`
//[unprefixed]~? ERROR incorrect value `auth-traps` for codegen option `pointer-authentication`
//[all_unknown]~? ERROR incorrect value `+I,+do,-not,-exist` for codegen option `pointer-authentication`
//[mixed]~? ERROR incorrect value `+elf-got,-imaginary` for codegen option `pointer-authentication`
