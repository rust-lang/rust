// compile-flags: -C overflow-checks=on

// EMIT_MIR return_place.add.ConstProp.diff
// EMIT_MIR return_place.add.PreCodegen.before.mir
fn add() -> u32 {
    2 + 2
}

fn main() {
    add();
}
