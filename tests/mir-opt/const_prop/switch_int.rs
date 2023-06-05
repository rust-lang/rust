// unit-test: ConstProp
// compile-flags: -Zmir-enable-passes=+SimplifyConstCondition-after-const-prop
// ignore-wasm32 compiled with panic=abort by default
#[inline(never)]
fn foo(_: i32) { }

// EMIT_MIR switch_int.main.ConstProp.diff
// EMIT_MIR switch_int.main.SimplifyConstCondition-after-const-prop.diff
fn main() {
    match 1 {
        1 => foo(0),
        _ => foo(-1),
    }
}
