// compile-flags: -C no-prepopulate-passes -O -Z mir-opt-level=2 -Zunsound-mir-opts

// Ensure that `x?` has no overhead on `Result<T, E>` due to identity `match`es in lowering.
// This requires inlining to trigger the MIR optimizations in `SimplifyArmIdentity`.

#![crate_type = "lib"]

type R = Result<u64, i32>;

#[no_mangle]
fn try_identity(x: R) -> R {
// CHECK: start:
// CHECK-NOT: br {{.*}}
// CHECK ret void
    let y = x?;
    Ok(y)
}
