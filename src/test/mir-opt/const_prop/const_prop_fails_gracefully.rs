#[inline(never)]
fn read(_: usize) { }

// EMIT_MIR rustc.main.ConstProp.diff
fn main() {
    const FOO: &i32 = &1;
    let x = FOO as *const i32 as usize;
    read(x);
}
