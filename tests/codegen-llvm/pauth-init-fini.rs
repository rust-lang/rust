// ignore-tidy-linelength
//@ only-aarch64-unknown-linux-pauthtest
//@ revisions: O0_PAUTH O3_PAUTH

//@ [O0_PAUTH] needs-llvm-components: aarch64
//@ [O0_PAUTH] compile-flags: --target=aarch64-unknown-linux-pauthtest -C opt-level=0
//@ [O3_PAUTH] needs-llvm-components: aarch64
//@ [O3_PAUTH] compile-flags: --target=aarch64-unknown-linux-pauthtest -C opt-level=3

// Make sure that init/fini metadata uses correct discriminator: 0xd9d4/55764

#![crate_type = "lib"]
#![feature(linkage)]

// O0_PAUTH: @{{[0-9A-Za-z_]+}}GLOBAL_INIT = constant ptr ptrauth (ptr @{{[0-9A-Za-z_]+}}init_fn, i32 0, i64 55764, ptr inttoptr (i64 1 to ptr)), section ".init_array.90"
// O3_PAUTH: @{{[0-9A-Za-z_]+}}GLOBAL_INIT = constant ptr ptrauth (ptr @{{[0-9A-Za-z_]+}}init_fn, i32 0, i64 55764, ptr inttoptr (i64 1 to ptr)), section ".init_array.90"
#[used]
#[link_section = ".init_array.90"]
static GLOBAL_INIT: extern "C" fn() = init_fn;

// O0_PAUTH: @{{[0-9A-Za-z_]+}}GLOBAL_FINI = constant ptr ptrauth (ptr @{{[0-9A-Za-z_]+}}fini_fn, i32 0, i64 55764, ptr inttoptr (i64 1 to ptr)), section ".fini_array.90"
// O3_PAUTH: @{{[0-9A-Za-z_]+}}GLOBAL_FINI = constant ptr ptrauth (ptr @{{[0-9A-Za-z_]+}}fini_fn, i32 0, i64 55764, ptr inttoptr (i64 1 to ptr)), section ".fini_array.90"
#[used]
#[link_section = ".fini_array.90"]
static GLOBAL_FINI: extern "C" fn(i32) = fini_fn;

extern "C" fn init_fn() {}
extern "C" fn fini_fn(_: i32) {}
