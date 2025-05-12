// Regression test for #50411: the MIR inliner was causing problems
// here because it would inline promoted code (which had already had
// elaborate-drops invoked on it) and then try to elaboate drops a
// second time. Uncool.

//@ compile-flags:-Zmir-opt-level=4
//@ build-pass

fn main() {
    let _ = (0 .. 1).filter(|_| [1].iter().all(|_| true)).count();
}
