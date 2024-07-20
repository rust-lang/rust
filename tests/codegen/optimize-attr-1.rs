//@ revisions: no-opt size-opt speed-opt
//@[no-opt] compile-flags: -Copt-level=0 -Ccodegen-units=1
//@[size-opt] compile-flags: -Copt-level=s -Ccodegen-units=1
//@[speed-opt] compile-flags: -Copt-level=3 -Ccodegen-units=1

#![feature(optimize_attribute)]
#![crate_type = "rlib"]

// CHECK-LABEL: define{{.*}}i32 @nothing
// CHECK-SAME: [[NOTHING_ATTRS:#[0-9]+]]
// CHECK-SIZE-OPT: ret i32 4
// CHECK-SPEED-OPT: ret i32 4
#[no_mangle]
pub fn nothing() -> i32 {
    2 + 2
}

// CHECK-LABEL: define{{.*}}i32 @size
// CHECK-SAME: [[SIZE_ATTRS:#[0-9]+]]
// CHECK-SIZE-OPT: ret i32 6
// CHECK-SPEED-OPT: ret i32 6
#[optimize(size)]
#[no_mangle]
pub fn size() -> i32 {
    3 + 3
}

// CHECK-LABEL: define{{.*}}i32 @speed
// CHECK-NO-OPT-SAME: [[NOTHING_ATTRS]]
// CHECK-SPEED-OPT-SAME: [[NOTHING_ATTRS]]
// CHECK-SIZE-OPT-SAME: [[SPEED_ATTRS:#[0-9]+]]
// CHECK-SIZE-OPT: ret i32 8
// CHECK-SPEED-OPT: ret i32 8
#[optimize(speed)]
#[no_mangle]
pub fn speed() -> i32 {
    4 + 4
}

// CHECK-NO-OPT-DAG: attributes [[SIZE_ATTRS]] = {{.*}}minsize{{.*}}optsize{{.*}}
// CHECK-SPEED-OPT-DAG: attributes [[SIZE_ATTRS]] = {{.*}}minsize{{.*}}optsize{{.*}}
// CHECK-SIZE-OPT-DAG: attributes [[NOTHING_ATTRS]] = {{.*}}optsize{{.*}}
// CHECK-SIZE-OPT-DAG: attributes [[SIZE_ATTRS]] = {{.*}}minsize{{.*}}optsize{{.*}}

// CHECK-SIZE-OPT: attributes [[SPEED_ATTRS]]
// CHECK-SIZE-OPT-NOT: minsize
// CHECK-SIZE-OPT-NOT: optsize
