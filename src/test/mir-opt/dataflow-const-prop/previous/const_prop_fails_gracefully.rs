// unit-test: DataflowConstProp
#[inline(never)]
fn read(_: usize) {}

// EMIT_MIR const_prop_fails_gracefully.main.DataflowConstProp.diff
fn main() {
    const FOO: &i32 = &1;
    let x = FOO as *const i32 as usize;
    read(x);
}
