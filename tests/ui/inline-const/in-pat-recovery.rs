// While `feature(inline_const_pat)` has been removed from the
// compiler, we should still make sure that the resulting error
// message is acceptable.
fn main() {
    match 1 {
        const { 1 + 7 } => {}
        //~^ ERROR `inline_const_pat` has been removed
        2 => {}
        _ => {}
    }
}
