// Issue #825: Should recheck the loop contition after continuing
fn main() {
    let i = 1;
    while i > 0 {
        assert (i > 0);
        log_full(core::debug, i);
        i -= 1;
        cont;
    }
}
