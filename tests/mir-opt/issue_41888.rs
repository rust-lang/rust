// skip-filecheck
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// check that we clear the "ADT master drop flag" even when there are
// no fields to be dropped.

// EMIT_MIR issue_41888.main.ElaborateDrops.diff
fn main() {
    let e;
    if cond() {
        e = E::F(K);
        if let E::F(_k) = e {
            // older versions of rustc used to not clear the
            // drop flag for `e` in this path.
        }
    }
}

fn cond() -> bool {
    false
}

struct K;

enum E {
    F(K),
    G(Box<E>),
}
