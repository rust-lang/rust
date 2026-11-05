// ignore-tidy-file-linelength
//@ add-minicore
// Run it at O0, so that the compiler doesn't optimise the calls away.
//@ revisions: DISC NO_DISC

//@ [DISC] needs-llvm-components: aarch64
//@ [DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=+function-pointer-type-discrimination -C opt-level=0
//@ [NO_DISC] needs-llvm-components: aarch64
//@ [NO_DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=-function-pointer-type-discrimination -C opt-level=0

// A function pointer is signed with the discriminator of the function instance it points to: for
// a generic function, the instantiated signature; for methods, the method's own signature. This
// holds whether the pointer is created at runtime or in a constant.
//
// Discriminators: `fn(i32)` = 2712, `fn(*const u8)` = 10942, `fn(i32) -> i32` = 2981.

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::Sync;

extern "C" fn generic<T>(_: T) {}

pub struct S;

impl S {
    extern "C" fn method(x: i32) -> i32 {
        x
    }
}

pub trait Tr {
    extern "C" fn assoc(x: i32) -> i32;
}

impl Tr for S {
    extern "C" fn assoc(x: i32) -> i32 {
        x
    }
}

pub struct Table {
    a: extern "C" fn(i32),
    b: extern "C" fn(*const u8),
    c: extern "C" fn(i32) -> i32,
}

impl Sync for Table {}

// DISC-DAG: @{{.*}}TABLE = constant <{ ptr, ptr, ptr }> <{ ptr ptrauth (ptr @{{.*}}generic{{.*}}, i32 0, i64 2712), ptr ptrauth (ptr @{{.*}}generic{{.*}}, i32 0, i64 10942), ptr ptrauth (ptr @{{.*}}assoc{{.*}}, i32 0, i64 2981) }>
// NO_DISC-DAG: @{{.*}}TABLE = constant <{ ptr, ptr, ptr }> <{ ptr ptrauth (ptr @{{.*}}generic{{.*}}, i32 0), ptr ptrauth (ptr @{{.*}}generic{{.*}}, i32 0), ptr ptrauth (ptr @{{.*}}assoc{{.*}}, i32 0) }>
#[used]
pub static TABLE: Table = Table { a: generic::<i32>, b: generic::<*const u8>, c: <S as Tr>::assoc };

// CHECK-LABEL: @reify_generic_i32(
#[no_mangle]
pub fn reify_generic_i32() -> extern "C" fn(i32) {
    // DISC: ret ptr ptrauth (ptr @{{.*}}generic{{.*}}, i32 0, i64 2712)
    // NO_DISC: ret ptr ptrauth (ptr @{{.*}}generic{{.*}}, i32 0)
    generic::<i32>
}

// CHECK-LABEL: @reify_generic_ptr(
#[no_mangle]
pub fn reify_generic_ptr() -> extern "C" fn(*const u8) {
    // DISC: ret ptr ptrauth (ptr @{{.*}}generic{{.*}}, i32 0, i64 10942)
    // NO_DISC: ret ptr ptrauth (ptr @{{.*}}generic{{.*}}, i32 0)
    generic::<*const u8>
}

// CHECK-LABEL: @reify_inherent_method(
#[no_mangle]
pub fn reify_inherent_method() -> extern "C" fn(i32) -> i32 {
    // DISC: ret ptr ptrauth (ptr @{{.*}}method{{.*}}, i32 0, i64 2981)
    // NO_DISC: ret ptr ptrauth (ptr @{{.*}}method{{.*}}, i32 0)
    S::method
}

// CHECK-LABEL: @reify_trait_method(
#[no_mangle]
pub fn reify_trait_method() -> extern "C" fn(i32) -> i32 {
    // DISC: ret ptr ptrauth (ptr @{{.*}}assoc{{.*}}, i32 0, i64 2981)
    // NO_DISC: ret ptr ptrauth (ptr @{{.*}}assoc{{.*}}, i32 0)
    <S as Tr>::assoc
}
