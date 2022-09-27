// unit-test: DataflowConstProp
// EMIT_MIR reify_fn_ptr.main.DataflowConstProp.diff

fn main() {
    let _ = main as usize as *const fn();
}
