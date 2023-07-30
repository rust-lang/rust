// compile-flags: -Copt-level=3 -Cdebug-assertions=true

// CHECK-LABEL: @slow_2_u(
#[no_mangle]
fn slow_2_u(a: u32) -> u32 {
    // CHECK: %_3 = icmp ult i32 %a, 32
    // CHECK-NEXT: br i1 %_3, label %bb1, label %panic, !prof !{{[0-9]+}}
    // CHECK-EMPTY:
    // CHECK-NEXT: bb1:
    // CHECK-NEXT: %_01 = shl nuw i32 1, %a
    // CHECK-NEXT: ret i32 %_0
    // CHECK-EMPTY:
    // CHECK-NEXT: panic:
    2u32.pow(a)
}

fn main() {
    slow_2_u(2);
}
