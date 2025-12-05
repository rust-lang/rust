// skip-filecheck
//@ test-mir-pass: Derefer
// EMIT_MIR derefer_test.main.Derefer.diff
fn main() {
    let mut a = (42, 43);
    let mut b = (99, &mut a);
    let x = &mut (*b.1).0;
    let y = &mut (*b.1).1;
}
