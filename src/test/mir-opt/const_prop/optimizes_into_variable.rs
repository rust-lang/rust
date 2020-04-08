// compile-flags: -C overflow-checks=on

struct Point {
    x: u32,
    y: u32,
}

// EMIT_MIR_FOR_EACH_BIT_WIDTH
// EMIT_MIR rustc.main.ConstProp.diff
// EMIT_MIR rustc.main.SimplifyLocals.after.mir
fn main() {
    let x = 2 + 2;
    let y = [0, 1, 2, 3, 4, 5][3];
    let z = (Point { x: 12, y: 42}).y;
}
