struct Droppy;

impl Drop for Droppy {
    fn drop(&mut self) {
        println!("drop");
    }
}

// EMIT_MIR logical_and_in_conditional.test_and.built.after.mir
fn test_and(a: i32, b: i32, c: i32) {
    if {
        let _t = Droppy;
        a == 0
    } && {
        let _t = Droppy;
        b == 0
    } && {
        let _t = Droppy;
        c == 0
    } {}
}

fn main() {
    for a in 0..1 {
        for b in 0..1 {
            for c in 0..1 {
                test_and(a, b, c);
            }
        }
    }
}
