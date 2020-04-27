#[inline(never)]
fn noop() {}

// EMIT_MIR rustc.main.ConstProp.diff
fn main() {
    match { let x = false; x } {
        true => noop(),
        false => {},
    }
}
