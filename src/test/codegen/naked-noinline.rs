// Checks that naked functions are never inlined.
// compile-flags: -O -Zmir-opt-level=2
// ignore-wasm32
#![crate_type = "lib"]
#![feature(asm)]
#![feature(naked_functions)]

#[inline(always)]
#[naked]
#[no_mangle]
pub unsafe extern "C" fn f() {
// Check that f has naked and noinline attributes.
//
// CHECK:       define void @f() unnamed_addr [[ATTR:#[0-9]+]]
// CHECK-NEXT:  start:
// CHECK-NEXT:    call void asm
    asm!("", options(noreturn));
}

#[no_mangle]
pub unsafe fn g() {
// Check that call to f is not inlined.
//
// CHECK-LABEL: define void @g()
// CHECK-NEXT:  start:
// CHECK-NEXT:    call void @f()
    f();
}

// CHECK: attributes [[ATTR]] = { naked noinline{{.*}} }
