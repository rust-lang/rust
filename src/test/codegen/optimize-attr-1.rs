// revisions: NO-OPT SIZE-OPT SPEED-OPT
//[NO-OPT] compile-flags: -Copt-level=0 -Ccodegen-units=1
//[SIZE-OPT] compile-flags: -Copt-level=s -Ccodegen-units=1
//[SPEED-OPT] compile-flags: -Copt-level=3 -Ccodegen-units=1

#![feature(optimize_attribute)]
#![crate_type="rlib"]

// CHECK-LABEL: define i32 @nothing
// CHECK-SAME: [[NOTHING_ATTRS:#[0-9]+]]
// NO-OPT: ret i32 %1
// SIZE-OPT: ret i32 4
// SPEEC-OPT: ret i32 4
#[no_mangle]
pub fn nothing() -> i32 {
    2 + 2
}

// CHECK-LABEL: define i32 @size
// CHECK-SAME: [[SIZE_ATTRS:#[0-9]+]]
// NO-OPT: ret i32 %1
// SIZE-OPT: ret i32 6
// SPEED-OPT: ret i32 6
#[optimize(size)]
#[no_mangle]
pub fn size() -> i32 {
    3 + 3
}

// CHECK-LABEL: define i32 @speed
// NO-OPT-SAME: [[NOTHING_ATTRS]]
// SPEED-OPT-SAME: [[NOTHING_ATTRS]]
// SIZE-OPT-SAME: [[SPEED_ATTRS:#[0-9]+]]
// NO-OPT: ret i32 %1
// SIZE-OPT: ret i32 8
// SPEED-OPT: ret i32 8
#[optimize(speed)]
#[no_mangle]
pub fn speed() -> i32 {
    4 + 4
}

// NO-OPT-DAG: attributes [[SIZE_ATTRS]] = {{.*}}minsize{{.*}}optsize{{.*}}
// SPEED-OPT-DAG: attributes [[SIZE_ATTRS]] = {{.*}}minsize{{.*}}optsize{{.*}}
// SIZE-OPT-DAG: attributes [[NOTHING_ATTRS]] = {{.*}}optsize{{.*}}
// SIZE-OPT-DAG: attributes [[SIZE_ATTRS]] = {{.*}}minsize{{.*}}optsize{{.*}}

// SIZE-OPT: attributes [[SPEED_ATTRS]]
// SIZE-OPT-NOT: minsize
// SIZE-OPT-NOT: optsize
