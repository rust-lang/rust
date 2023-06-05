// unit-test: ConstProp
// EMIT_MIR reify_fn_ptr.main.ConstProp.diff

fn main() {
    let _ = main as usize as *const fn();
}
