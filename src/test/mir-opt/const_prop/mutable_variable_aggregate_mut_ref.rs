// compile-flags: -O

// EMIT_MIR rustc.main.ConstProp.diff
fn main() {
    let mut x = (42, 43);
    let z = &mut x;
    z.1 = 99;
    let y = x;
}
