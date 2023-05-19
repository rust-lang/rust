struct Droppy(bool);

impl Drop for Droppy {
    fn drop(&mut self) {
        println!("drop");
    }
}

// EMIT_MIR logical_and_in_conditional.test_and.built.after.mir
fn test_and(a: i32, b: i32, c: i32) {
    if Droppy(a == 0).0 && Droppy(b == 0).0 && Droppy(c == 0).0 {}
}

// https://github.com/rust-lang/rust/issues/111583
// EMIT_MIR logical_and_in_conditional.function_from_issue_111583.built.after.mir
fn function_from_issue_111583(z: f64) {
    let mask = (1 << 38) - 1;
    let mut ret = 0;
    if (z.to_bits() >> 8) & mask == 0 && z % 0.0625 < 1e-13 {
        ret += 1;
    }
}

fn main() {
    function_from_issue_111583(0.0);
    for a in 0..1 {
        for b in 0..1 {
            for c in 0..1 {
                test_and(a, b, c);
            }
        }
    }
}
