//@ revisions: 32BIT 64BIT
//@ compile-flags: -O
//@ min-llvm-version: 18.0
//@ [32BIT] only-32bit
//@ [64BIT] only-64bit

#![crate_type = "lib"]

// Confirm that the `assume` calls from the length allows LLVM to know that some
// math on the indices can be done without overflow risk.

// CHECK-LABEL: @slice_length_demo
#[no_mangle]
pub unsafe fn slice_length_demo(x: &[u16]) {
    // 32BIT: %[[LIMIT:.+]] = icmp ult [[USIZE:i32]] %x.1, [[#0x40000000]]
    // 64BIT: %[[LIMIT:.+]] = icmp ult [[USIZE:i64]] %x.1, [[#0x4000000000000000]]
    // CHECK: tail call void @llvm.assume(i1 %[[LIMIT]])

    // CHECK: %[[Y:.+]] = phi [[USIZE]]
    // CHECK-SAME: [ 0, %start ]
    // CHECK: %[[PLUS_ONE:.+]] = add nuw nsw [[USIZE]] %[[Y]], 1
    // CHECK: call void @do_something([[USIZE]] noundef %[[PLUS_ONE]])
    // CHECK: %[[TIMES_TWO:.+]] = shl nuw nsw [[USIZE]] %[[Y]], 1
    // CHECK: call void @do_something([[USIZE]] noundef %[[TIMES_TWO]])
    for y in 0..x.len() {
        do_something(y + 1);
        do_something(y * 2);
    }
}

// CHECK-LABEL: @nested_slice_length
#[no_mangle]
pub unsafe fn nested_slice_length(x: &[f32], y: &[f32]) {
    // 32BIT: %[[LIMIT:.+]] = icmp ult [[USIZE:i32]] %x.1, [[#0x20000000]]
    // 64BIT: %[[LIMIT:.+]] = icmp ult [[USIZE:i64]] %x.1, [[#0x2000000000000000]]
    // CHECK: tail call void @llvm.assume(i1 %[[LIMIT]])
    // 32BIT: %[[LIMIT:.+]] = icmp ult [[USIZE]] %y.1, [[#0x20000000]]
    // 64BIT: %[[LIMIT:.+]] = icmp ult [[USIZE]] %y.1, [[#0x2000000000000000]]
    // CHECK: tail call void @llvm.assume(i1 %[[LIMIT]])

    // CHECK: %[[J:.+]] = phi [[USIZE]]
    // CHECK: %[[I:.+]] = phi [[USIZE]]
    // CHECK-NOT: phi
    // CHECK: %[[T:.+]] = add nuw nsw [[USIZE]] %[[I]], %[[J]]
    // CHECK: call void @do_something([[USIZE]] noundef %[[T]])
    for i in 0..x.len() {
        for j in 0..y.len() {
            do_something(i + j);
        }
    }
}

extern {
    fn do_something(x: usize);
}
