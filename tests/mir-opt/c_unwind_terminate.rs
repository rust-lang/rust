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
    // CHECK: drop
    // CHECK-SAME: unwind: [[unwind:bb.*]]]
    // CHECK: [[unwind]] (cleanup)
    // CHECK-NEXT: terminate(abi)
    let _val = Noise;
    panic();
}

fn main() {}
