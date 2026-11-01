//@ test-mir-pass: GVN

// Verify that we do not propagate the contents of this mutable static.
static mut STATIC: u32 = 0x42424242;

// EMIT_MIR mutable_variable_no_prop.main.GVN.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: debug x => [[x:_.*]];
    // CHECK: debug y => [[y:_.*]];
    // CHECK: [[x]] = const 42_u32;
    // CHECK: [[tmp:_.*]] = copy (*{{_.*}});
    // CHECK: [[x]] = move [[tmp]];
    // CHECK: [[y]] = copy [[x]];
    let mut x = 42;
    unsafe {
        x = STATIC;
    }
    let y = x;
}
