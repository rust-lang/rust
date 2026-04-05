// ignore-tidy-linelength
//@ only-aarch64-unknown-linux-pauthtest
//@ revisions: O0_PAUTH O3_PAUTH

//@ [O0_PAUTH] needs-llvm-components: aarch64
//@ [O0_PAUTH] compile-flags: --target=aarch64-unknown-linux-pauthtest -C opt-level=0
//@ [O3_PAUTH] needs-llvm-components: aarch64
//@ [O3_PAUTH] compile-flags: --target=aarch64-unknown-linux-pauthtest -C opt-level=3

// Make sure that direct extern "C" calls are not handled by pointer authentication operand bundle
// logic.
use std::ffi::c_void;
use std::hint::black_box;

extern "C" {
    fn rand() -> i32;
    fn add(a: i32, b: i32) -> i32;
    fn sub(a: i32, b: i32) -> i32;

    // Corresponds to: void *woof;
    static mut woof: *mut c_void;
    fn direct_function_taking_void_arg(data: *mut c_void);
    fn direct_no_arg();
    fn direct_function_taking_fp_arg(func: unsafe extern "C" fn());
}

type CFnPtr = unsafe extern "C" fn(i32, i32) -> i32;

// CHECK-LABE: test_indirect_call
#[inline(never)]
fn test_indirect_call() {
    let fp_add: CFnPtr = black_box(add);
    let fp_sub: CFnPtr = black_box(sub);

    unsafe {
        // O0_PAUTH: call {{.*}}i32 %fp_add({{.*}}) #{{[0-9]+}} [ "ptrauth"(i32 0, i64 0) ]
        // O3_PAUTH: call {{.*}}i32 %fp_add({{.*}}) #{{[0-9]+}} [ "ptrauth"(i32 0, i64 0) ]
        let _id1 = fp_add(7, 4);
        // O0_PAUTH: call {{.*}}i32 %fp_sub({{.*}}) #{{[0-9]+}} [ "ptrauth"(i32 0, i64 0) ]
        // O3_PAUTH: call {{.*}}i32 %fp_sub({{.*}}) #{{[0-9]+}} [ "ptrauth"(i32 0, i64 0) ]
        let _id2 = fp_sub(10, 6);
    }

    // Also test calling via conditional pointer
    unsafe {
        // O0_PAUTH: call {{.*}}i32 ptrauth (ptr @rand, i32 0)({{.*}}) #{{[0-9]+}} [ "ptrauth"(i32 0, i64 0) ]
        // O3_PAUTH: call {{.*}}i32  @rand() #
        let use_add = rand() % 2 == 0;
        // O0_PAUTH: store ptr ptrauth (ptr @sub, i32 0), ptr %[[FP_O0:[a-zA-Z0-9_.]+]]
        // O0_PAUTH: store ptr ptrauth (ptr @add, i32 0), ptr %[[FP_O0]]{{.*}}
        // O0_PAUTH: %[[LOAD_FP_O0:[a-zA-Z0-9_.]+]] = load ptr, ptr %[[FP_O0]]{{.*}}
        // O3_PAUTH: %[[FP_O3:.*]] = select i1 %{{.*}}, ptr ptrauth (ptr @add, i32 0), ptr ptrauth (ptr @sub, i32 0)
        let fp: CFnPtr = if use_add { add } else { sub };
        // O0_PAUTH: call i32 %[[LOAD_FP_O0]](i32 1, i32 2) #{{[0-9]+}} [ "ptrauth"(i32 0, i64 0) ]
        // O3_PAUTH: call {{.*}}i32 %[[FP_O3]](i32 noundef 1, i32 noundef 2) #{{[0-9]+}} [ "ptrauth"(i32 0, i64 0) ]
        let _id3 = fp(1, 2);
    }

    unsafe {
        direct_function_taking_fp_arg(direct_no_arg);
    }
}

// CHECK-LABE: test_direct_call
#[inline(never)]
fn test_direct_call() {
    unsafe {
        // O0_PAUTH: call {{.*}}i32 ptrauth (ptr @add, i32 0)({{.*}}) #{{[0-9]+}} [ "ptrauth"(i32 0, i64 0) ]
        // O3_PAUTH: call {{.*}}i32 @add(i32 {{.*}}2, i32 {{.*}}3) #
        let _d1 = add(2, 3);
        // O0_PAUTH: call {{.*}}i32 ptrauth (ptr @sub, i32 0)({{.*}}) #{{[0-9]+}} [ "ptrauth"(i32 0, i64 0) ]
        // O3_PAUTH: call {{.*}}i32 @sub(i32 {{.*}}5, i32 {{.*}}1) #
        let _d2 = sub(5, 1);

        // O0_PAUTH: call {{.*}}void ptrauth (ptr @direct_function_taking_void_arg, i32 0)({{.*}}) #{{[0-9]+}} [ "ptrauth"(i32 0, i64 0) ]
        // O3_PAUTH: {{(tail )?}}call void @direct_function_taking_void_arg(ptr noundef %{{.*}}) #
        direct_function_taking_void_arg(woof);
    }
}

fn main() {
    test_indirect_call();
    test_direct_call();
}

// O0_PAUTH: !{{[0-9]+}} = !{i32 7, !"ptrauth-sign-personality", i32 1}
// O3_PAUTH: !{{[0-9]+}} = !{i32 7, !"ptrauth-sign-personality", i32 1}
