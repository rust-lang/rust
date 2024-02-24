//@ compile-flags: -C opt-level=3

#![crate_type = "lib"]

fn infinite_loop() -> u8 {
    let i = 2;
    while i > 1 {}
    1
}

// CHECK-LABEL: @test
#[no_mangle]
fn test() -> u8 {
    // CHECK-NOT: unreachable
    // CHECK: br label %{{.+}}
    // CHECK-NOT: unreachable
    let x = infinite_loop();
    x
}
