// ignore-tidy-file-linelength
//@ add-minicore
//@ revisions: DISC NO_DISC

//@ [DISC] needs-llvm-components: aarch64
//@ [DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=+function-pointer-type-discrimination -C opt-level=0
//@ [NO_DISC] needs-llvm-components: aarch64
//@ [NO_DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=-function-pointer-type-discrimination -C opt-level=0

// `#[ptrauth_c_discriminator(N)]` marks the slots of a static that hold a code address with an
// explicit encoding (`N`):
// * on a field, that field in every static containing it;
// * on a static, the static itself or every element of an array static.
//
// A code address in a marked slot is signed with `N`; every other one keeps the discriminator of
// its function. Matching is by offset, so null entries, arrays and nested structs need nothing
// special.
//
// Without type discrimination, every code address is signed with 0.
//
// Discriminators:
// * `extern "C" fn(i32)` = 2712,
// * `extern "C" fn()` = 18983.

#![feature(no_core, lang_items, ptrauth_c_discriminator)]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::Option::{self, None, Some};
use minicore::mem::transmute;

pub type GCallback = Option<unsafe extern "C" fn()>;

pub extern "C" fn cb_i32(_: i32) {}
pub extern "C" fn cb_void() {}

// A static that is itself a slot.
#[used]
#[ptrauth_c_discriminator(0)]
// DISC-DAG: @{{.*}}OPAQUE = {{.*}}ptr ptrauth (ptr @{{[0-9A-Za-z_]*}}cb_i32, i32 0), align 8
// NO_DISC-DAG: @{{.*}}OPAQUE = {{.*}}ptr ptrauth (ptr @{{[0-9A-Za-z_]*}}cb_i32, i32 0), align 8
static mut OPAQUE: *const u8 = cb_i32 as *const u8;

// The same, unmarked: the function's own discriminator.
#[used]
// DISC-DAG: @{{.*}}UNMARKED = {{.*}}ptr ptrauth (ptr @{{[0-9A-Za-z_]*}}cb_i32, i32 0, i64 2712), align 8
// NO_DISC-DAG: @{{.*}}UNMARKED = {{.*}}ptr ptrauth (ptr @{{[0-9A-Za-z_]*}}cb_i32, i32 0), align 8
static mut UNMARKED: *const u8 = cb_i32 as *const u8;

// A stati carray: every element.
#[used]
#[ptrauth_c_discriminator(0)]
// DISC-DAG: @{{.*}}TABLE = {{.*}}ptr ptrauth (ptr @{{[0-9A-Za-z_]*}}cb_i32, i32 0), ptr ptrauth (ptr @{{[0-9A-Za-z_]*}}cb_void, i32 0) }>
// NO_DISC-DAG: @{{.*}}TABLE = {{.*}}ptr ptrauth (ptr @{{[0-9A-Za-z_]*}}cb_i32, i32 0), ptr ptrauth (ptr @{{[0-9A-Za-z_]*}}cb_void, i32 0) }>
static mut TABLE: [*const u8; 2] = [cb_i32 as *const u8, cb_void as *const u8];

// A type erased method table: one marked field, in an array of structs.
#[repr(C)]
pub struct NativeMethod {
    pub name: *const u8,
    #[ptrauth_c_discriminator(0)]
    pub fn_ptr: *const u8,
}

#[used]
// DISC-DAG: @{{.*}}METHODS = {{.*}}ptr ptrauth (ptr @{{[0-9A-Za-z_]*}}cb_i32, i32 0), {{.*}}ptr ptrauth (ptr @{{[0-9A-Za-z_]*}}cb_void, i32 0) }>
// NO_DISC-DAG: @{{.*}}METHODS = {{.*}}ptr ptrauth (ptr @{{[0-9A-Za-z_]*}}cb_i32, i32 0), {{.*}}ptr ptrauth (ptr @{{[0-9A-Za-z_]*}}cb_void, i32 0) }>
static mut METHODS: [NativeMethod; 2] = [
    NativeMethod { name: 0 as *const u8, fn_ptr: cb_i32 as *const u8 },
    NativeMethod { name: 0 as *const u8, fn_ptr: cb_void as *const u8 },
];

// Exactly typed, generic callback and `void *` slots in one struct. The typed field needs no
// mark: the function's own discriminator is the one of its type.
#[repr(C)]
pub struct Mixed {
    pub typed: Option<extern "C" fn(i32)>,
    #[ptrauth_c_discriminator(18983)]
    pub callback: GCallback,
    #[ptrauth_c_discriminator(0)]
    pub user: *const u8,
}

#[used]
// DISC-DAG: @{{.*}}MIXED = {{.*}}<{ ptr ptrauth (ptr @{{[0-9A-Za-z_]*}}cb_i32, i32 0, i64 2712), ptr ptrauth (ptr @{{[0-9A-Za-z_]*}}cb_i32, i32 0, i64 18983), ptr ptrauth (ptr @{{[0-9A-Za-z_]*}}cb_i32, i32 0) }>
// NO_DISC-DAG: @{{.*}}MIXED = {{.*}}<{ ptr ptrauth (ptr @{{[0-9A-Za-z_]*}}cb_i32, i32 0), ptr ptrauth (ptr @{{[0-9A-Za-z_]*}}cb_i32, i32 0), ptr ptrauth (ptr @{{[0-9A-Za-z_]*}}cb_i32, i32 0) }>
static mut MIXED: Mixed = Mixed {
    typed: Some(cb_i32),
    callback: unsafe { transmute::<extern "C" fn(i32), GCallback>(cb_i32) },
    user: cb_i32 as *const u8,
};

// Null entries do not shift anything: the code address in `user` is still matched by its offset.
#[used]
// DISC-DAG: @{{.*}}SPARSE = {{.*}}ptr ptrauth (ptr @{{[0-9A-Za-z_]*}}cb_void, i32 0) }>
// NO_DISC-DAG: @{{.*}}SPARSE = {{.*}}ptr ptrauth (ptr @{{[0-9A-Za-z_]*}}cb_void, i32 0) }>
static mut SPARSE: Mixed = Mixed { typed: None, callback: None, user: cb_void as *const u8 };

// A marked field of a nested struct.
#[repr(C)]
pub struct Outer {
    pub tag: usize,
    pub inner: NativeMethod,
}

#[used]
// DISC-DAG: @{{.*}}NESTED = {{.*}}ptr ptrauth (ptr @{{[0-9A-Za-z_]*}}cb_void, i32 0) }>
// NO_DISC-DAG: @{{.*}}NESTED = {{.*}}ptr ptrauth (ptr @{{[0-9A-Za-z_]*}}cb_void, i32 0) }>
static mut NESTED: Outer =
    Outer { tag: 1, inner: NativeMethod { name: 0 as *const u8, fn_ptr: cb_void as *const u8 } };

// A union: the marked member's encoding applies to the slot, whichever member initialised it.
#[repr(C)]
pub union MethodPtr {
    pub with_args: Option<extern "C" fn(i32)>,
    #[ptrauth_c_discriminator(18983)]
    pub erased: GCallback,
}

#[used]
// DISC-DAG: @{{.*}}UNION = {{.*}}ptr ptrauth (ptr @{{[0-9A-Za-z_]*}}cb_i32, i32 0, i64 18983), align 8
// NO_DISC-DAG: @{{.*}}UNION = {{.*}}ptr ptrauth (ptr @{{[0-9A-Za-z_]*}}cb_i32, i32 0), align 8
static mut UNION: MethodPtr = MethodPtr { with_args: Some(cb_i32) };
