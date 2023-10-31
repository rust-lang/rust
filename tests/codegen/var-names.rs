// compile-flags: -O -C no-prepopulate-passes

#![crate_type = "lib"]

// CHECK-LABEL: define{{.*}}i32 @test(i32 noundef %a, i32 noundef %b)
#[no_mangle]
pub fn test(a: u32, b: u32) -> u32 {
    let c = a + b;
    // CHECK: %c = add i32 %a, %b
    let d = c;
    let e = d * a;
    // CHECK-NEXT: %e = mul i32 %c, %a
    e
    // CHECK-NEXT: ret i32 %e
}
