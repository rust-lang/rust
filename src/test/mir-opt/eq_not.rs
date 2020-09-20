// EMIT_MIR eq_not.opt_simple.InstCombine.diff
fn opt_simple(x: u8) {
    assert!(x == 2);
}

// EMIT_MIR eq_not.opt_has_storage.InstCombine.diff
fn opt_has_storage(x: Vec<u8>) {
    assert!(x.len() == 2);
}

// EMIT_MIR eq_not.opt_has_later_use.InstCombine.diff
fn opt_has_later_use(x: Vec<u8>) -> u8 {
    assert!(x.len() == 2);
    x[0]
}

fn main() {
    opt_simple(0);
    opt_has_storage(vec![]);
    opt_has_later_use(vec![]);
}
