// EMIT_MIR and_star.opt_mutable.InstCombine.diff
fn opt_mutable(dst: &mut [u8]) {
    dst.get_mut(0);
}

// EMIT_MIR and_star.opt_immutable.InstCombine.diff
fn opt_immutable(dst: &[u8]) {
    dst.get(0);
}

fn main() {
    opt_mutable(&mut [1]);
    opt_immutable(&[1])
}
