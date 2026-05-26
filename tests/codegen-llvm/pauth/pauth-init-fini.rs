//@ add-minicore
// ignore-tidy-linelength
//@ only-pauthtest
//@ revisions: O0_PAUTH O3_PAUTH O0_PAUTH-ADDR-DISC O3_PAUTH-ADDR-DISC O0_PAUTH-NO-INIT-FINI O3_PAUTH-NO-INIT-FINI

//@ [O0_PAUTH] needs-llvm-components: aarch64
//@ [O0_PAUTH] compile-flags: --target=aarch64-unknown-linux-pauthtest -C opt-level=0
//@ [O0_PAUTH-ADDR-DISC] needs-llvm-components: aarch64
//@ [O0_PAUTH-ADDR-DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest -C opt-level=0 -Zpointer-authentication=+init-fini-address-discrimination
//@ [O0_PAUTH-NO-INIT-FINI] needs-llvm-components: aarch64
//@ [O0_PAUTH-NO-INIT-FINI] compile-flags: --target=aarch64-unknown-linux-pauthtest -C opt-level=0 -Zpointer-authentication=-init-fini
//@ [O3_PAUTH] needs-llvm-components: aarch64
//@ [O3_PAUTH] compile-flags: --target=aarch64-unknown-linux-pauthtest -C opt-level=3
//@ [O3_PAUTH-ADDR-DISC] needs-llvm-components: aarch64
//@ [O3_PAUTH-ADDR-DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest -C opt-level=3 -Zpointer-authentication=+init-fini-address-discrimination
//@ [O3_PAUTH-NO-INIT-FINI] needs-llvm-components: aarch64
//@ [O3_PAUTH-NO-INIT-FINI] compile-flags: --target=aarch64-unknown-linux-pauthtest -C opt-level=0 -Zpointer-authentication=-init-fini

// Make sure that init/fini metadata uses correct discriminator: 0xd9d4/55764 - ptrauth_string_discriminator("init_fini").
// And that address discriminator can be enabled.

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

// O0_PAUTH: @{{[0-9A-Za-z_]+}}GLOBAL_INIT = constant ptr ptrauth (ptr @{{[0-9A-Za-z_]+}}init_fn, i32 0, i64 55764, ptr inttoptr (i64 1 to ptr)), section ".init_array.90"
// O3_PAUTH: @{{[0-9A-Za-z_]+}}GLOBAL_INIT = constant ptr ptrauth (ptr @{{[0-9A-Za-z_]+}}init_fn, i32 0, i64 55764, ptr inttoptr (i64 1 to ptr)), section ".init_array.90"
// O0_PAUTH-ADDR-DISC: @{{[0-9A-Za-z_]+}}GLOBAL_INIT = constant ptr ptrauth (ptr @{{[0-9A-Za-z_]+}}init_fn, i32 0, i64 55764, ptr @_RNvCsf7kshQi9mOB_15pauth_init_fini7init_fn), section ".init_array.90"
// O3_PAUTH-ADDR-DISC: @{{[0-9A-Za-z_]+}}GLOBAL_INIT = constant ptr ptrauth (ptr @{{[0-9A-Za-z_]+}}init_fn, i32 0, i64 55764, ptr @_RNvCsf7kshQi9mOB_15pauth_init_fini7init_fn), section ".init_array.90"
// O0_PAUTH-NO-INIT-FINI-NOT: @{{[0-9A-Za-z_]+}}GLOBAL_INIT = constant ptr ptrauth
// O0_PAUTH-NO-INIT-FINI-ADDR-DISC: @{{[0-9A-Za-z_]+}}GLOBAL_INIT = constant ptr ptrauth
#[used]
#[link_section = ".init_array.90"]
static GLOBAL_INIT: extern "C" fn() = init_fn;

// O0_PAUTH: @{{[0-9A-Za-z_]+}}GLOBAL_FINI = constant ptr ptrauth (ptr @{{[0-9A-Za-z_]+}}fini_fn, i32 0, i64 55764, ptr inttoptr (i64 1 to ptr)), section ".fini_array.90"
// O3_PAUTH: @{{[0-9A-Za-z_]+}}GLOBAL_FINI = constant ptr ptrauth (ptr @{{[0-9A-Za-z_]+}}fini_fn, i32 0, i64 55764, ptr inttoptr (i64 1 to ptr)), section ".fini_array.90"
// O0_PAUTH-ADDR-DISC: @{{[0-9A-Za-z_]+}}GLOBAL_FINI = constant ptr ptrauth (ptr @{{[0-9A-Za-z_]+}}fini_fn, i32 0, i64 55764, ptr @_RNvCsf7kshQi9mOB_15pauth_init_fini7fini_fn), section ".fini_array.90"
// O3_PAUTH-ADDR-DISC: @{{[0-9A-Za-z_]+}}GLOBAL_FINI = constant ptr ptrauth (ptr @{{[0-9A-Za-z_]+}}fini_fn, i32 0, i64 55764, ptr @_RNvCsf7kshQi9mOB_15pauth_init_fini7fini_fn), section ".fini_array.90"
// O0_PAUTH-NO-INIT-FINI-NOT: @{{[0-9A-Za-z_]+}}GLOBAL_FINI = constant ptr ptrauth
// O3_PAUTH-NO-INIT-FINI-NOT: @{{[0-9A-Za-z_]+}}GLOBAL_FINI = constant ptr ptrauth
#[used]
#[link_section = ".fini_array.90"]
static GLOBAL_FINI: extern "C" fn(i32) = fini_fn;

extern "C" fn init_fn() {}
extern "C" fn fini_fn(_: i32) {}
