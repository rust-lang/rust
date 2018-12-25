// ignore-tidy-linelength

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
// CHECK-NEXT: define void @naked_with_args(i{{[0-9]+}})
pub fn naked_with_args(a: isize) {
    // CHECK-NEXT: {{.+}}:
    // CHECK-NEXT: %a = alloca i{{[0-9]+}}
    &a; // keep variable in an alloca
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
// CHECK-NEXT: define i{{[0-9]+}} @naked_with_args_and_return(i{{[0-9]+}})
#[no_mangle]
#[naked]
pub fn naked_with_args_and_return(a: isize) -> isize {
    // CHECK-NEXT: {{.+}}:
    // CHECK-NEXT: %a = alloca i{{[0-9]+}}
    &a; // keep variable in an alloca
    // CHECK: ret i{{[0-9]+}} %{{[0-9]+}}
    a
}

// CHECK: Function Attrs: naked
// CHECK-NEXT: define void @naked_recursive()
#[no_mangle]
#[naked]
pub fn naked_recursive() {
    // CHECK-NEXT: {{.+}}:
    // CHECK-NEXT: call void @naked_empty()

    // FIXME(#39685) Avoid one block per call.
    // CHECK-NEXT: br label %bb1
    // CHECK: bb1:

    naked_empty();

    // CHECK-NEXT: %{{[0-9]+}} = call i{{[0-9]+}} @naked_with_return()

    // FIXME(#39685) Avoid one block per call.
    // CHECK-NEXT: br label %bb2
    // CHECK: bb2:

    // CHECK-NEXT: %{{[0-9]+}} = call i{{[0-9]+}} @naked_with_args_and_return(i{{[0-9]+}} %{{[0-9]+}})

    // FIXME(#39685) Avoid one block per call.
    // CHECK-NEXT: br label %bb3
    // CHECK: bb3:

    // CHECK-NEXT: call void @naked_with_args(i{{[0-9]+}} %{{[0-9]+}})

    // FIXME(#39685) Avoid one block per call.
    // CHECK-NEXT: br label %bb4
    // CHECK: bb4:

    naked_with_args(
        naked_with_args_and_return(
            naked_with_return()
        )
    );
    // CHECK-NEXT: ret void
}
