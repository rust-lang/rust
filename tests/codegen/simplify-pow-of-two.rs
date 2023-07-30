// compile-flags: -Copt-level=3

// CHECK-LABEL: @slow_2_u(
#[no_mangle]
fn slow_2_u(a: u32) -> u32 {
    // CHECK: %_3 = icmp ult i32 %a, 32
    // CHECK-NEXT: %_5 = zext i1 %_3 to i32
    // CHECK-NEXT: %0 = and i32 %a, 31
    // CHECK-NEXT: %_01 = shl nuw i32 %_5, %0
    // CHECK-NEXT: ret i32 %_01
    2u32.pow(a)
}

fn main() {
    slow_2_u(2);
}
