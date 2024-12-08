// skip-filecheck
//@ test-mir-pass: Derefer
// EMIT_MIR derefer_test_multiple.main.Derefer.diff
fn main() {
    let mut a = (42, 43);
    let mut b = (99, &mut a);
    let mut c = (11, &mut b);
    let mut d = (13, &mut c);
    let x = &mut (*d.1).1.1.1;
    let y = &mut (*d.1).1.1.1;
}
