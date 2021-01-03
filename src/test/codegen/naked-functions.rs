// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(naked_functions)]

// CHECK: Function Attrs: naked
// CHECK-NEXT: define void @naked_empty()
#[no_mangle]
#[naked]
pub fn naked_empty() {
    // CHECK-NEXT: {{.+}}:
    // CHECK-NEXT: ret void
}

// CHECK: Function Attrs: naked
#[no_mangle]
#[naked]
// CHECK-NEXT: define void @naked_with_args(i{{[0-9]+( %a)?}})
pub fn naked_with_args(a: isize) {
    // CHECK-NEXT: {{.+}}:
    // CHECK: ret void
}

// CHECK: Function Attrs: naked
// CHECK-NEXT: define i{{[0-9]+}} @naked_with_return()
#[no_mangle]
#[naked]
pub fn naked_with_return() -> isize {
    // CHECK-NEXT: {{.+}}:
    // CHECK-NEXT: ret i{{[0-9]+}} 0
    0
}

// CHECK: Function Attrs: naked
// CHECK-NEXT: define i{{[0-9]+}} @naked_with_args_and_return(i{{[0-9]+( %a)?}})
#[no_mangle]
#[naked]
pub fn naked_with_args_and_return(a: isize) -> isize {
    // CHECK-NEXT: {{.+}}:
    // CHECK: ret i{{[0-9]+}} 0
    0
}
