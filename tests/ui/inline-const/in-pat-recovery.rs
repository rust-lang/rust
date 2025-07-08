// While `feature(inline_const_pat)` has been removed from the
// compiler, we should still make sure that the resulting error
// message is acceptable.
fn main() {
    match 1 {
        const { 1 + 7 } => {}
        //~^ ERROR const blocks cannot be used as patterns
        2 => {}
        _ => {}
    }
}
