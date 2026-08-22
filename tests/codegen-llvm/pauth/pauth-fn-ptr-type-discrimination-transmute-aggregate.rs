// ignore-tidy-file-linelength
//@ add-minicore
//@ only-pauthtest
// Run it at O0, so that the compiler doesn't optimise the calls away.
//@ revisions: DISC NO_DISC

//@ [DISC] needs-llvm-components: aarch64
//@ [DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=+function-pointer-type-discrimination -C opt-level=0
//@ [NO_DISC] needs-llvm-components: aarch64
//@ [NO_DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=-function-pointer-type-discrimination -C opt-level=0

// Test generation of function-pointer type discriminators. No C code for this test as strict
// aliasing rules would not allow for pointer casting. The alternative - using memcpy in clang does
// not generate resigning.
//
// Transmute aggregates containing function pointers.
//
// Equivalent C version at the bottom of the file. Some Rust aggregates (for example, tuples) do
// not have direct C equivalents. For ease of IR inspection, the C types retain the Rust-inspired
// naming, even where the data structures do not correspond exactly.

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]

extern crate minicore;

use minicore::mem::transmute;

extern "C" fn f() {}
extern "C" fn g(_: i32) {}

#[repr(transparent)]
struct Wrapper(extern "C" fn(i32));

#[repr(C)]
struct StructA {
    fp: extern "C" fn(i32),
}

#[repr(C)]
struct StructB {
    fp: extern "C" fn(),
}

// DISC-NOT: llvm.ptrauth.resign

// CHECK-LABEL: test_transparent_wrapper
pub fn test_transparent_wrapper() {
    unsafe {
        let w = Wrapper(g);
        // DISC: [[TRANSMUTED:%.*]] = call i64 @llvm.ptrauth.resign(i64 ptrtoint (ptr ptrauth (ptr @{{.*}}g, i32 0, i64 2712) to i64), i32 0, i64 2712, i32 0, i64 18983)
        let p: extern "C" fn() = transmute(w);
        // DISC: [[P:%.*]] = inttoptr i64 [[TRANSMUTED]] to ptr
        // DISC: call void [[P]]() #[[#]] [ "ptrauth"(i32 0, i64 18983) ]
        // NO_DISC: call void ptrauth (ptr @{{.*}}g, i32 0)() #[[#]] [ "ptrauth"(i32 0, i64 0) ]
        p();
    }
}

// CHECK-LABEL: test_tuple
pub fn test_tuple() {
    unsafe {
        // DISC: store ptr ptrauth (ptr @{{.*}}g, i32 0, i64 2712), ptr [[ALLOCA_1:%.*]]
        let t = (g as extern "C" fn(i32),);
        // DISC: call i64 @llvm.ptrauth.resign(i64 %{{.*}}, i32 0, i64 2712, i32 0, i64 18983)
        let u: (extern "C" fn(),) = transmute(t);
        // DISC: call void %{{.*}}() #[[#]] [ "ptrauth"(i32 0, i64 18983) ]
        // NO_DISC: call void ptrauth (ptr @{{.*}}g, i32 0)() #[[#]] [ "ptrauth"(i32 0, i64 0) ]
        (u.0)();
    }
}

// CHECK-LABEL: test_tuple_two_entries
pub fn test_tuple_two_entries() {
    // This test performs two resigns. Track the sequence of aggregate accesses to ensure each
    // tuple's field is resigned with the discriminator corresponding to its new function pointer
    // type after the transmute.
    unsafe {
        // DISC: [[DST:%.*]] = alloca [16 x i8]
        // Set up source tuple (t).
        // DISC: [[SRC:%.*]] = alloca [16 x i8]
        // DISC: store ptr ptrauth (ptr @{{.*}}g, i32 0, i64 2712), ptr [[SRC]]
        // DISC: [[SRC1:%.*]] = getelementptr inbounds i8, ptr [[SRC]], i64 8
        // DISC: store ptr ptrauth (ptr @{{.*}}f, i32 0, i64 18983), ptr [[SRC1]]
        let t = (g as extern "C" fn(i32), f as extern "C" fn());

        // Field 0: ld - resigned - st
        // DISC: [[LOAD0:%.*]] = load ptr, ptr [[SRC]]
        // DISC: [[INT0:%.*]] = ptrtoint ptr [[LOAD0]] to i64
        // DISC: [[RESIGN0:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[INT0]], i32 0, i64 2712, i32 0, i64 18983)
        // DISC: [[PTR0:%.*]] = inttoptr i64 [[RESIGN0]] to ptr
        // DISC: store ptr [[PTR0]], ptr [[DST]]
        // Field 1: ld - resigned - st
        // DISC: [[SRC1_AGAIN:%.*]] = getelementptr inbounds i8, ptr [[SRC]], i64 8
        // DISC: [[DST1:%.*]] = getelementptr inbounds i8, ptr [[DST]], i64 8
        // DISC: [[LOAD1:%.*]] = load ptr, ptr [[SRC1_AGAIN]]
        // DISC: [[INT1:%.*]] = ptrtoint ptr [[LOAD1]] to i64
        // DISC: [[RESIGN1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[INT1]], i32 0, i64 18983, i32 0, i64 2712)
        // DISC: [[PTR1:%.*]] = inttoptr i64 [[RESIGN1]] to ptr
        // DISC: store ptr [[PTR1]], ptr [[DST1]]
        let u: (extern "C" fn(), extern "C" fn(i32)) = transmute(t);
        // Reload both
        // DISC: [[U0:%.*]] = load ptr, ptr [[DST]]
        // DISC: [[DST1_AGAIN:%.*]] = getelementptr inbounds i8, ptr [[DST]], i64 8
        // DISC: [[U1:%.*]] = load ptr, ptr [[DST1_AGAIN]]
        // Calls
        // DISC: call void [[U0]]() #[[#]] [ "ptrauth"(i32 0, i64 18983) ]
        // NO_DISC: call void ptrauth (ptr @{{.*}}g, i32 0)() #[[#]] [ "ptrauth"(i32 0, i64 0) ]
        (u.0)();
        // DISC: call void [[U1]](i32 42) #[[#]] [ "ptrauth"(i32 0, i64 2712) ]
        // NO_DISC: call void ptrauth (ptr @{{.*}}f, i32 0)(i32 42) #[[#]] [ "ptrauth"(i32 0, i64 0) ]
        (u.1)(42);
    }
}

// CHECK-LABEL: test_nested_tuple
pub fn test_nested_tuple() {
    unsafe {
        // DISC: store ptr ptrauth (ptr @{{.*}}g, i32 0, i64 2712)
        let t = ((g as extern "C" fn(i32),),);
        // DISC: call i64 @llvm.ptrauth.resign(i64 %{{.*}}, i32 0, i64 2712, i32 0, i64 18983)
        let u: ((extern "C" fn(),),) = transmute(t);
        // DISC: call void %{{.*}}() #[[#]] [ "ptrauth"(i32 0, i64 18983) ]
        // NO_DISC: call void ptrauth (ptr @{{.*}}g, i32 0)() #[[#]] [ "ptrauth"(i32 0, i64 0) ]
        (u.0).0();
    }
}

// CHECK-LABEL: test_nested_tuple_with_sibling
pub fn test_nested_tuple_with_sibling() {
    // The nested tuple is flattened in LLVM IR, but both leaf function pointers must still be
    // resigned according to their new types.
    // The filecheck directives are quite verbose, but we must track the flow of each field
    // independently.
    unsafe {
        // DISC: [[DST:%.*]] = alloca [16 x i8]
        // DISC: [[SRC:%.*]] = alloca [16 x i8]
        // Source field 0: g
        // DISC: store ptr ptrauth (ptr @{{.*}}g, i32 0, i64 2712), ptr [[SRC]]
        // Source field 1: f
        // DISC: [[SRC1:%.*]] = getelementptr inbounds i8, ptr [[SRC]], i64 8
        // DISC: store ptr ptrauth (ptr @{{.*}}f, i32 0, i64 18983), ptr [[SRC1]]
        let t = ((g as extern "C" fn(i32),), f as extern "C" fn());

        // g's transmute
        // DISC: [[LOAD0:%.*]] = load ptr, ptr [[SRC]]
        // DISC: [[INT0:%.*]] = ptrtoint ptr [[LOAD0]] to i64
        // DISC: [[RESIGN0:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[INT0]], i32 0, i64 2712, i32 0, i64 18983)
        // DISC: [[PTR0:%.*]] = inttoptr i64 [[RESIGN0]] to ptr
        // DISC: store ptr [[PTR0]], ptr [[DST]]
        // f's transmute
        // DISC: [[SRC1_AGAIN:%.*]] = getelementptr inbounds i8, ptr [[SRC]], i64 8
        // DISC: [[DST1:%.*]] = getelementptr inbounds i8, ptr [[DST]], i64 8
        // DISC: [[LOAD1:%.*]] = load ptr, ptr [[SRC1_AGAIN]]
        // DISC: [[INT1:%.*]] = ptrtoint ptr [[LOAD1]] to i64
        // DISC: [[RESIGN1:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[INT1]], i32 0, i64 18983, i32 0, i64 2712)
        // DISC: [[PTR1:%.*]] = inttoptr i64 [[RESIGN1]] to ptr
        // DISC: store ptr [[PTR1]], ptr [[DST1]]
        let u: ((extern "C" fn(),), extern "C" fn(i32)) = transmute(t);
        // Reload
        // DISC: [[U0:%.*]] = load ptr, ptr [[DST]]
        // DISC: [[DST1_AGAIN:%.*]] = getelementptr inbounds i8, ptr [[DST]], i64 8
        // DISC: [[U1:%.*]] = load ptr, ptr [[DST1_AGAIN]]
        // Calls
        // DISC: call void [[U0]]() #[[#]] [ "ptrauth"(i32 0, i64 18983) ]
        // NO_DISC: call void ptrauth (ptr @_{{.*}}g, i32 0)() #[[#]] [ "ptrauth"(i32 0, i64 0) ]
        (u.0).0();
        // DISC: call void [[U1]](i32 42) #[[#]] [ "ptrauth"(i32 0, i64 2712) ]
        // NO_DISC: call void ptrauth (ptr @{{.*}}f, i32 0)(i32 42) #[[#]] [ "ptrauth"(i32 0, i64 0) ]
        (u.1)(42);
    }
}

// CHECK-LABEL: test_struct
pub fn test_struct() {
    unsafe {
        // DISC: store ptr ptrauth (ptr @{{.*}}g, i32 0, i64 2712)
        // NO_DISC: store ptr ptrauth (ptr @{{.*}}g, i32 0)
        let a = StructA { fp: g };
        // DISC: call i64 @llvm.ptrauth.resign(i64 %{{.*}}, i32 0, i64 2712, i32 0, i64 18983)
        // NO_DISC: call void @llvm.memcpy.p0.p0.i64
        let b: StructB = transmute(a);
        // DISC: call void %{{.*}}() #[[#]] [ "ptrauth"(i32 0, i64 18983) ]
        (b.fp)();
        // NO_DISC: call void %{{.*}}() #[[#]] [ "ptrauth"(i32 0, i64 0) ]
    }
}

// CHECK-LABEL: test_array
pub fn test_array() {
    unsafe {
        // DISC: store ptr ptrauth (ptr @{{.*}}g, i32 0, i64 2712)
        // DISC: store ptr ptrauth (ptr @{{.*}}g, i32 0, i64 2712)
        // NO_DISC: store ptr ptrauth (ptr @{{.*}}g, i32 0)
        // NO_DISC: store ptr ptrauth (ptr @{{.*}}g, i32 0)
        let arr: [extern "C" fn(i32); 2] = [g, g];
        // DISC: call i64 @llvm.ptrauth.resign(i64 %{{.*}}, i32 0, i64 2712, i32 0, i64 18983)
        // DISC: call i64 @llvm.ptrauth.resign(i64 %{{.*}}, i32 0, i64 2712, i32 0, i64 18983)
        // NO_DISC: call void @llvm.memcpy.p0.p0.i64
        let arr2: [extern "C" fn(); 2] = transmute(arr);

        let [p, q] = arr2;
        // DISC: call void %{{.*}}() #[[#]] [ "ptrauth"(i32 0, i64 18983) ]
        // NO_DISC: call void %{{.*}}() #[[#]] [ "ptrauth"(i32 0, i64 0) ]
        p();
        // DISC: call void %{{.*}}() #[[#]] [ "ptrauth"(i32 0, i64 18983) ]
        // NO_DISC: call void %{{.*}}() #[[#]] [ "ptrauth"(i32 0, i64 0) ]
        q();
    }
}

#[repr(transparent)]
struct SrcRef(&'static extern "C" fn(i32));

#[repr(transparent)]
struct DstRef(&'static extern "C" fn());

// CHECK-LABEL: test_ref
pub fn test_ref(g: &'static extern "C" fn(i32)) -> DstRef {
    // References are not transmuted. The output is the same for DISC and NO_DISC.
    // CHECK: ret ptr %{{.*}}g
    unsafe { transmute(SrcRef(g)) }
}

struct Pair<T>(T, u8);

// CHECK-LABEL: test_aggregate_wrapper
pub fn test_aggregate_wrapper(g: extern "C" fn(i32)) -> Pair<extern "C" fn()> {
    // The pair is copied field-by-field.
    // DISC: store ptr %{{.*}}g, ptr [[FN_PTR:%.*]]

    // The function pointer field is resigned with the new discriminator.
    // DISC: [[FN_PTR_LD:%.*]] = load ptr, ptr [[FN_PTR]]
    // DISC: [[INT:%.*]] = ptrtoint ptr [[FN_PTR_LD]] to i64
    // DISC: [[TRANSMUTED:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[INT]], i32 0, i64 2712, i32 0, i64 18983)
    // DISC: [[PTR_TRANS:%.*]] = inttoptr i64 [[TRANSMUTED]] to ptr
    // DISC: store ptr [[PTR_TRANS]], ptr {{.*}}

    // The non-pointer field is copied as a plain byte.
    // DISC: load i8, ptr {{.*}}
    // DISC: store i8 {{.*}}, ptr {{.*}}

    // Returning the aggregate.
    // DISC: ret { ptr, i8 }

    // The aggregate must not be copied as a raw blob.
    // DISC-NOT: llvm.memcpy

    // NO_DISC: insertvalue { ptr, i8 } poison, ptr %{{.*}}g, 0
    // NO_DISC: insertvalue { ptr, i8 } %{{.*}}, i8 42, 1
    unsafe { transmute(Pair(g, 42)) }
}

struct NonTransparent<T>(T);

// CHECK-LABEL: test_nontransparent
pub fn test_nontransparent(g: extern "C" fn(i32)) -> NonTransparent<extern "C" fn()> {
    unsafe {
        // DISC: store ptr %{{.*}}g, ptr [[SRC]]

        // Field is loaded from aggregate and resigned
        // DISC: [[LOAD:%.*]] = load ptr, ptr [[SRC]]
        // DISC: [[INT:%.*]] = ptrtoint ptr [[LOAD]] to i64
        // DISC: [[RESIGN:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[INT]], i32 0, i64 2712, i32 0, i64 18983)
        // DISC: [[PTR:%.*]] = inttoptr i64 [[RESIGN]] to ptr

        // Store transformed aggregate field
        // DISC: store ptr [[PTR]], ptr %{{.*}}

        // Result is called with destination discriminator
        // DISC: [[RES:%.*]] = load ptr, ptr %{{.*}}
        // DISC: call void [[RES]]() #{{[0-9]+}} [ "ptrauth"(i32 0, i64 18983) ]

        // In NO_DISC transmute becomes a noop.
        // NO_DISC: call void %{{.*}}g() #[[#]] [ "ptrauth"(i32 0, i64 0) ]
        // NO_DISC: ret ptr %{{.*}}g
        let res: NonTransparent<extern "C" fn()> = transmute(NonTransparent(g));

        (res.0)();
        res
    }
}

// Equivalent C sample:
// #include <stdbool.h>
//
// typedef void (*fn0_t)(void);
// typedef void (*fn1_t)(int);
//
// void f(void) {}
// void g(int x) { (void)x; }
//
// struct Wrapper {
//   fn1_t value;
// };
//
// struct StructA {
//   fn1_t fp;
// };
//
// struct StructB {
//   fn0_t fp;
// };
//
// struct Tuple1Src {
//   fn1_t v0;
// };
//
// struct Tuple1Dst {
//   fn0_t v0;
// };
//
// struct Tuple2Src {
//   fn1_t v0;
//   fn0_t v1;
// };
//
// struct Tuple2Dst {
//   fn0_t v0;
//   fn1_t v1;
// };
//
// struct NestedSrc {
//   struct Tuple1Src v0;
// };
//
// struct NestedDst {
//   struct Tuple1Dst v0;
// };
//
// struct NestedSiblingSrc {
//   struct Tuple1Src v0;
//   fn0_t v1;
// };
//
// struct NestedSiblingDst {
//   struct Tuple1Dst v0;
//   fn1_t v1;
// };
//
// static fn1_t F = g;
//
// struct SrcRef {
//   fn1_t *value;
// };
//
// struct DstRef {
//   fn0_t *value;
// };
//
// struct PairDst {
//   fn0_t fp;
//   unsigned char value;
// };
//
// struct PairSrc {
//   fn1_t fp;
//   unsigned char value;
// };
//
// struct NonTransparentSrc {
//   fn1_t value;
// };
//
// struct NonTransparentDst {
//   fn0_t value;
// };
//
// void test_transparent_wrapper(void) {
//   struct Wrapper w = {g};
//
//   fn0_t p = (fn0_t)w.value;
//
//   p();
// }
//
// void test_tuple(void) {
//   struct Tuple1Src t = {g};
//
//   struct Tuple1Dst u = {(fn0_t)t.v0};
//
//   u.v0();
// }
//
// void test_tuple_two_entries(void) {
//   struct Tuple2Src t = {g, f};
//
//   struct Tuple2Dst u = {(fn0_t)t.v0, (fn1_t)t.v1};
//
//   u.v0();
//   u.v1(42);
// }
//
// void test_nested_tuple(void) {
//   struct NestedSrc t = {{g}};
//
//   struct NestedDst u = {{(fn0_t)t.v0.v0}};
//
//   u.v0.v0();
// }
//
// void test_nested_tuple_with_sibling(void) {
//   struct NestedSiblingSrc t = {{g}, f};
//
//   struct NestedSiblingDst u = {{(fn0_t)t.v0.v0}, (fn1_t)t.v1};
//
//   u.v0.v0();
//   u.v1(42);
// }
//
// void test_struct(void) {
//   struct StructA a = {g};
//
//   struct StructB b = {(fn0_t)a.fp};
//
//   b.fp();
// }
//
// void test_array(void) {
//   fn1_t src[2] = {g, g};
//
//   fn0_t dst[2] = {(fn0_t)src[0], (fn0_t)src[1]};
//
//   dst[0]();
//   dst[1]();
// }
//
// struct DstRef test_ref(fn1_t *p) { return (struct DstRef){(fn0_t *)p}; }
//
// struct PairDst test_aggregate_wrapper(fn1_t p) {
//   struct PairSrc src = {p, 42};
//
//   return (struct PairDst){(fn0_t)src.fp, src.value};
// }
//
// struct NonTransparentDst test_nontransparent(fn1_t p) {
//   struct NonTransparentSrc src = {p};
//
//   struct NonTransparentDst dst = {(fn0_t)src.value};
//
//   dst.value();
//
//   return dst;
// }
