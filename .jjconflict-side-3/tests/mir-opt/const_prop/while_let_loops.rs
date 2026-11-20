//@ test-mir-pass: GVN
// EMIT_MIR while_let_loops.change_loop_body.GVN.diff

pub fn change_loop_body() {
    // CHECK-LABEL: fn change_loop_body(
    // CHECK: switchInt(const 0_isize)
    let mut _x = 0;
    while let Some(0u32) = None {
        _x = 1;
        break;
    }
}

fn main() {
    change_loop_body();
}
