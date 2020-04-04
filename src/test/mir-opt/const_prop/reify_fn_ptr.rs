// EMIT_MIR rustc.main.ConstProp.diff

fn main() {
    let _ = main as usize as *const fn();
}
