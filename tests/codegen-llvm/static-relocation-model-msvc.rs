// Verify linkage of external symbols in the static relocation model on MSVC.
//
//@ compile-flags: -Copt-level=3 -C relocation-model=static
//@ aux-build: extern_decl.rs
//@ only-x86_64-pc-windows-msvc

#![crate_type = "rlib"]

extern crate extern_decl;

// The `extern_decl` definitions are imported from a statically linked rust
// crate, thus they are expected to be marked `dso_local` without `dllimport`.
//
// The `access_extern()` symbol is from this compilation unit, thus we expect
// it to be marked `dso_local` as well, given the static relocation model.
//
// CHECK: @extern_static = external dso_local local_unnamed_addr global i8
// CHECK: define dso_local noundef i8 @access_extern() {{.*}}
// CHECK: declare dso_local noundef i8 @extern_fn() {{.*}}

#[no_mangle]
pub fn access_extern() -> u8 {
    unsafe { extern_decl::extern_fn() + extern_decl::extern_static }
}
