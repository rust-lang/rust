#[inline(never)]
fn foo(_: i32) { }

// EMIT_MIR rustc.main.ConstProp.diff
// EMIT_MIR rustc.main.SimplifyBranches-after-const-prop.diff
fn main() {
    match 1 {
        1 => foo(0),
        _ => foo(-1),
    }
}
