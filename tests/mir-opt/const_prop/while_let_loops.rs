// unit-test: ConstProp
// EMIT_MIR while_let_loops.change_loop_body.ConstProp.diff

pub fn change_loop_body() {
    let mut _x = 0;
    while let Some(0u32) = None {
        _x = 1;
        break;
    }
}

fn main() {
    change_loop_body();
}
