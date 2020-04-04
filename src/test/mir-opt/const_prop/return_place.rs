// compile-flags: -C overflow-checks=on

// EMIT_MIR rustc.add.ConstProp.diff
// EMIT_MIR rustc.add.PreCodegen.before.mir
fn add() -> u32 {
    2 + 2
}

fn main() {
    add();
}
