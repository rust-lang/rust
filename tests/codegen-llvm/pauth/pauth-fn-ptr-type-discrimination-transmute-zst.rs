// ignore-tidy-file-linelength
//@ add-minicore
//@ only-pauthtest
// Run it at O0, so that the compiler doesn't optimise the calls away.
//@ revisions: DISC NO_DISC

//@ [DISC] needs-llvm-components: aarch64
//@ [DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=+function-pointer-type-discrimination -C opt-level=0
//@ [NO_DISC] needs-llvm-components: aarch64
//@ [NO_DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=-function-pointer-type-discrimination -C opt-level=0

// `PhantomData<T>` is 1-aligned/zero-sized for any `T`, so these are all valid `repr(transparent)`
// wrappers with the wrapped fn pointer *not* at field index 0. `WrappedN` wraps a *different* fn
// signature than `RootSrc.fN`, so a real resign is required.

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]

extern crate minicore;

use minicore::mem::transmute;
use minicore::{PhantomData, Sync, ptr};

#[repr(transparent)]
struct Wrapped0(PhantomData<u8>, extern "C" fn(i32));

#[repr(transparent)]
struct Wrapped1((), extern "C" fn(i64, i64));

#[repr(transparent)]
struct Wrapped2(PhantomData<u8>, PhantomData<u16>, extern "C" fn(i64, i64, f32));

#[repr(transparent)]
struct Wrapped3([(); 0], extern "C" fn());

pub struct RootSrc {
    f0: extern "C" fn(),
    f1: extern "C" fn(i32),
    f2: extern "C" fn(i64, i64),
    f3: extern "C" fn(i64, i64, f32),
}

pub struct RootDst {
    f0: Wrapped0,
    f1: Wrapped1,
    f2: Wrapped2,
    f3: Wrapped3,
}

impl Sync for RootSrc {}
impl Sync for RootDst {}

#[no_mangle]
// CHECK-LABEL-DAG: test_transparent_nonzero_field
pub fn test_transparent_nonzero_field(src: RootSrc) -> RootDst {
    // NO_DISC-NOT: call i64 @llvm.ptrauth.resign
    RootDst {
        // field 0: load -> resign(18983 -> 2712) -> store
        // DISC: [[SRC0:%.*]] = load ptr, ptr [[SRC:%.*]],
        // DISC: [[SRC0I:%.*]] = ptrtoint ptr [[SRC0]] to i64
        // DISC: [[RESIGN0:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[SRC0I]], i32 0, i64 18983, i32 0, i64 2712)
        // DISC: [[DST0:%.*]] = inttoptr i64 [[RESIGN0]] to ptr
        f0: unsafe { transmute::<extern "C" fn(), Wrapped0>(src.f0) },

        // field 1: load -> resign(2712 -> 55265) -> store
        // DISC: [[SRC1PTR:%.*]] = getelementptr inbounds i8, ptr [[SRC]], i64 8
        // DISC: [[SRC1:%.*]] = load ptr, ptr [[SRC1PTR]]
        // DISC: [[SRC1I:%.*]] = ptrtoint ptr [[SRC1]] to i64
        // DISC: [[RESIGN1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[SRC1I]], i32 0, i64 2712, i32 0, i64 55265)
        // DISC: [[DST1:%.*]] = inttoptr i64 [[RESIGN1]] to ptr
        f1: unsafe { transmute::<extern "C" fn(i32), Wrapped1>(src.f1) },

        // field 2: load -> resign(55265 -> 44485) -> store -- two leading 1-ZSTs this time
        // DISC: [[SRC2PTR:%.*]] = getelementptr inbounds i8, ptr [[SRC]], i64 16
        // DISC: [[SRC2:%.*]] = load ptr, ptr [[SRC2PTR]]
        // DISC: [[SRC2I:%.*]] = ptrtoint ptr [[SRC2]] to i64
        // DISC: [[RESIGN2:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[SRC2I]], i32 0, i64 55265, i32 0, i64 44485)
        // DISC: [[DST2:%.*]] = inttoptr i64 [[RESIGN2]] to ptr
        f2: unsafe { transmute::<extern "C" fn(i64, i64), Wrapped2>(src.f2) },

        // field 3: load -> resign(44485 -> 18983) -> store
        // DISC: [[SRC3PTR:%.*]] = getelementptr inbounds i8, ptr [[SRC]], i64 24
        // DISC: [[SRC3:%.*]] = load ptr, ptr [[SRC3PTR]]
        // DISC: [[SRC3I:%.*]] = ptrtoint ptr [[SRC3]] to i64
        // DISC: [[RESIGN3:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[SRC3I]], i32 0, i64 44485, i32 0, i64 18983)
        // DISC: [[DST3:%.*]] = inttoptr i64 [[RESIGN3]] to ptr
        f3: unsafe { transmute::<extern "C" fn(i64, i64, f32), Wrapped3>(src.f3) },
    }
}
