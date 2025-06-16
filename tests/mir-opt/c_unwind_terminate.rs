//@ needs-unwind

struct Noise;
impl Drop for Noise {
    fn drop(&mut self) {
        eprintln!("Noisy Drop");
    }
}

fn panic() {
    panic!();
}

// EMIT_MIR c_unwind_terminate.test.AbortUnwindingCalls.after.mir
extern "C" fn test() {
    // CHECK-LABEL: fn test(
    // CHECK: panic
    // CHECK-SAME: unwind: [[panic_unwind:bb.*]]]
    // CHECK: drop
    // CHECK-SAME: unwind: [[drop_unwind:bb.*]]]
    // CHECK: [[panic_unwind]] (cleanup)
    // CHECK-NEXT: drop
    // CHECK-SAME: terminate(cleanup)
    // CHECK: [[drop_unwind]] (cleanup)
    // CHECK-NEXT: terminate(abi)
    let _val = Noise;
    panic();
}

fn main() {}
