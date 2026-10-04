// ignore-tidy-file-linelength
//@ add-minicore
// Run it at O0, so that the compiler doesn't optimise the calls away.
//@ revisions: DISC NO_DISC

//@ [DISC] needs-llvm-components: aarch64
//@ [DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=+function-pointer-type-discrimination -C opt-level=0
//@ [NO_DISC] needs-llvm-components: aarch64
//@ [NO_DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=-function-pointer-type-discrimination -C opt-level=0

// Transmutes through raw pointers. Make sure we match clang's behavior of treating raw pointers as
// zero-discriminated.

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]

extern crate minicore;

use minicore::Option;
use minicore::Option::{None, Some};
use minicore::mem::transmute;

extern "C" fn f() {}
extern "C" fn g(_: i32) {}
// NO_DISC-NOT: = call i64 @llvm.ptrauth.resign

// CHECK-LABEL-DAG: test_fn_ptr_raw_ptr_fn_ptr
pub fn test_fn_ptr_raw_ptr_fn_ptr() {
    unsafe {
        // DISC: call i64 @llvm.ptrauth.resign(i64 ptrtoint (ptr ptrauth (ptr @{{.*}}g, i32 0, i64 2712) to i64), i32 0, i64 2712, i32 0, i64 0)
        let p: extern "C" fn(i32) = g;
        let raw: *const () = transmute(p);

        // DISC: = call i64 @llvm.ptrauth.resign(i64 %{{.*}}, i32 0, i64 0, i32 0, i64 2712)
        let q: extern "C" fn(i32) = transmute(raw);

        // DISC: call void %{{.*}}(i32 123) #[[#]] [ "ptrauth"(i32 0, i64 2712) ]
        // NO_DISC: call void ptrauth (ptr @{{.*}}g, i32 0)(i32 123) #[[#]] [ "ptrauth"(i32 0, i64 0) ]
        q(123);
    }
}

// CHECK-LABEL-DAG: test_round_trip_through_mutable_raw_ptr
pub fn test_round_trip_through_mutable_raw_ptr() {
    unsafe {
        // DISC: call i64 @llvm.ptrauth.resign(i64 ptrtoint (ptr ptrauth (ptr @{{.*}}f, i32 0, i64 18983) to i64), i32 0, i64 18983, i32 0, i64 0)
        let p: extern "C" fn() = f;
        let raw: *mut () = transmute(p);

        // DISC: call i64 @llvm.ptrauth.resign(i64 %{{.*}}, i32 0, i64 0, i32 0, i64 18983)
        let q: extern "C" fn() = transmute(raw);

        // DISC: call void %{{.*}}() #[[#]] [ "ptrauth"(i32 0, i64 18983) ]
        // NO_DISC: call void ptrauth (ptr @{{.*}}f, i32 0)() #[[#]] [ "ptrauth"(i32 0, i64 0) ]
        q();
    }
}

// CHECK-LABEL-DAG: test_transmute_raw_ptr_to_memory_backed_fn_ptr
pub fn test_transmute_raw_ptr_to_memory_backed_fn_ptr() {
    unsafe {
        // DISC: call i64 @llvm.ptrauth.resign(i64 ptrtoint (ptr ptrauth (ptr @{{.*}}g, i32 0, i64 2712) to i64), i32 0, i64 2712, i32 0, i64 0)
        let p: extern "C" fn(i32) = g;
        let raw: *const () = transmute(p);

        // DISC: call i64 @llvm.ptrauth.resign(i64 %{{.*}}, i32 0, i64 0, i32 0, i64 2712)
        let q: extern "C" fn(i32) = transmute(raw);
        // Taking `q`'s address forces it into a memory place rather than an SSA.
        let _addr_taken: *const _ = &q;

        // DISC: call void %{{.*}}(i32 123) #[[#]] [ "ptrauth"(i32 0, i64 2712) ]
        // NO_DISC: call void %{{.*}}(i32 123) #[[#]] [ "ptrauth"(i32 0, i64 0) ]
        q(123);
    }
}

#[repr(transparent)]
struct Wrapper(extern "C" fn(i32));

// CHECK-LABEL-DAG: test_transmute_raw_ptr_to_memory_backed_wrapped_fn_ptr
pub fn test_transmute_raw_ptr_to_memory_backed_wrapped_fn_ptr() {
    unsafe {
        // DISC: call i64 @llvm.ptrauth.resign(i64 ptrtoint (ptr ptrauth (ptr @{{.*}}g, i32 0, i64 2712) to i64), i32 0, i64 2712, i32 0, i64 0)
        let p: extern "C" fn(i32) = g;
        let raw: *const () = transmute(p);

        // DISC: call i64 @llvm.ptrauth.resign(i64 %{{.*}}, i32 0, i64 0, i32 0, i64 2712)
        let w: Wrapper = transmute(raw);
        // Force `w` into a memory place rather than an SSA.
        let _addr_taken: *const _ = &w;

        // DISC: call void %{{.*}}(i32 123) #[[#]] [ "ptrauth"(i32 0, i64 2712) ]
        // NO_DISC: call void %{{.*}}(i32 123) #[[#]] [ "ptrauth"(i32 0, i64 0) ]
        (w.0)(123);
    }
}

// CHECK-LABEL-DAG: test_transmute_raw_ptr_to_memory_backed_option_fn_ptr
pub fn test_transmute_raw_ptr_to_memory_backed_option_fn_ptr() {
    unsafe {
        // DISC: call i64 @llvm.ptrauth.resign(i64 ptrtoint (ptr ptrauth (ptr @{{.*}}g, i32 0, i64 2712) to i64), i32 0, i64 2712, i32 0, i64 0)
        let p: extern "C" fn(i32) = g;
        let raw: *const () = transmute(p);

        // DISC: icmp eq i64 %{{.*}}, 0
        // DISC: br i1 %{{.*}}, label %ptrauth.null, label %ptrauth.resign
        // DISC: ptrauth.resign:
        // DISC: call i64 @llvm.ptrauth.resign(i64 %{{.*}}, i32 0, i64 0, i32 0, i64 2712)
        let opt: Option<extern "C" fn(i32)> = transmute(raw);
        // Force `opt` into a memory place.
        let _addr_taken: *const _ = &opt;

        // DISC: call void %{{.*}}(i32 123) #[[#]] [ "ptrauth"(i32 0, i64 2712) ]
        // NO_DISC: call void %{{.*}}(i32 123) #[[#]] [ "ptrauth"(i32 0, i64 0) ]
        match opt {
            Some(f) => f(123),
            None => {}
        }
    }
}

// CHECK-LABEL-DAG: test_transmute_const_null_raw_ptr_to_memory_backed_option_fn_ptr
pub fn test_transmute_const_null_raw_ptr_to_memory_backed_option_fn_ptr() {
    unsafe {
        // Fast path, null store/load feeding straight into the match.
        let raw: *const () = 0 as *const ();

        // DISC: store ptr null, ptr %{{.*}}, align 8
        // DISC-NOT: call i64 @llvm.ptrauth.resign
        // DISC-NOT: ptrauth.null
        let opt: Option<extern "C" fn(i32)> = transmute(raw);
        // Force `opt` into a memory place rather than an SSA-representable local.
        let _addr_taken: *const _ = &opt;

        // DISC: call void %{{.*}}(i32 123) #[[#]] [ "ptrauth"(i32 0, i64 2712) ]
        // NO_DISC: call void %{{.*}}(i32 123) #[[#]] [ "ptrauth"(i32 0, i64 0) ]
        match opt {
            Some(f) => f(123),
            None => {}
        }
    }
}

// CHECK-LABEL-DAG: test_transmute_wrapped_fn_ptr_to_memory_backed_raw_ptr
pub fn test_transmute_wrapped_fn_ptr_to_memory_backed_raw_ptr() {
    unsafe {
        let w = Wrapper(g);

        // DISC: call i64 @llvm.ptrauth.resign(i64 ptrtoint (ptr ptrauth (ptr @{{.*}}g, i32 0, i64 2712) to i64), i32 0, i64 2712, i32 0, i64 0)
        let raw: *const () = transmute(w);
        // Force `raw` into a memory place.
        let _addr_taken: *const _ = &raw;

        // DISC: call i64 @llvm.ptrauth.resign(i64 %{{.*}}, i32 0, i64 0, i32 0, i64 2712)
        let f: extern "C" fn(i32) = transmute(raw);
        // DISC: call void %{{.*}}(i32 123) #[[#]] [ "ptrauth"(i32 0, i64 2712) ]
        // NO_DISC: call void %{{.*}}(i32 123) #[[#]] [ "ptrauth"(i32 0, i64 0) ]
        f(123);
    }
}
