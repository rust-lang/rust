// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// EMIT_MIR remove_unneeded_drops.opt.RemoveUnneededDrops.diff
fn opt(x: bool) {
    drop(x);
}

// EMIT_MIR remove_unneeded_drops.dont_opt.RemoveUnneededDrops.diff
fn dont_opt(x: Vec<bool>) {
    drop(x);
}

// EMIT_MIR remove_unneeded_drops.opt_generic_copy.RemoveUnneededDrops.diff
fn opt_generic_copy<T: Copy>(x: T) {
    drop(x);
}

// EMIT_MIR remove_unneeded_drops.cannot_opt_generic.RemoveUnneededDrops.diff
// since the pass is not running on monomorphisized code,
// we can't (but probably should) optimize this
fn cannot_opt_generic<T>(x: T) {
    drop(x);
}

fn main() {
    opt(true);
    opt_generic_copy(42);
    cannot_opt_generic(42);
    dont_opt(vec![true]);
}
