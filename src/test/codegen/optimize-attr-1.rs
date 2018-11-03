// revisions: NO-OPT SIZE-OPT SPEED-OPT
// [NO-OPT]compile-flags: -Copt-level=0
// [SIZE-OPT]compile-flags: -Copt-level=s
// [SPEED-OPT]compile-flags: -Copt-level=3

#![feature(optimize_attribute)]
#![crate_type="rlib"]

// NO-OPT: Function Attrs:{{.*}}optnone
// NO-OPT-NOT: {{optsize|minsize}}
// NO-OPT-NEXT: @nothing
// NO-OPT: ret i32 %1
//
// SIZE-OPT: Function Attrs:{{.*}}optsize
// SIZE-OPT-NOT: {{minsize|optnone}}
// SIZE-OPT-NEXT: @nothing
// SIZE-OPT-NEXT: start
// SIZE-OPT-NEXT: ret i32 4
//
// SPEED-OPT: Function Attrs:
// SPEED-OPT-NOT: {{minsize|optnone|optsize}}
// SPEED-OPT-NEXT: @nothing
// SPEED-OPT-NEXT: start
// SPEED-OPT-NEXT: ret i32 4
#[no_mangle]
pub fn nothing() -> i32 {
    2 + 2
}

// CHECK: Function Attrs:{{.*}} minsize{{.*}}optsize
// CHECK-NEXT: @size
// CHECK-NEXT: start
// CHECK-NEXT: ret i32 4
#[optimize(size)]
#[no_mangle]
pub fn size() -> i32 {
    2 + 2
}

// CHECK: Function Attrs:
// CHECK-NOT: {{minsize|optsize|optnone}}
// CHECK-NEXT: @speed
// CHECK-NEXT: start
// CHECK-NEXT: ret i32 4
#[optimize(speed)]
#[no_mangle]
pub fn speed() -> i32 {
    2 + 2
}
