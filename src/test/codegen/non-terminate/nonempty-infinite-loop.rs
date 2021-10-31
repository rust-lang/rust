// compile-flags: -C opt-level=3

#![crate_type = "lib"]

// Verify that we don't miscompile this even if rustc didn't apply the trivial loop detection to
// insert the sideeffect intrinsic.

fn infinite_loop() -> u8 {
    let mut x = 0;
    // CHECK-NOT: sideeffect
    loop {
        if x == 42 {
            x = 0;
        } else {
            x = 42;
        }
    }
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
