//@ run-pass
//
// Regression test for match lowering to MIR: when gathering candidates, by the time we get to the
// range we know the range will only match on the failure case of the switchint. Hence we mustn't
// add the `1` to the switchint or the range would be incorrectly sorted.
#![allow(unreachable_patterns)]
fn main() {
    match 1 {
        10 => unreachable!(),
        0..=5 => {}
        1 => unreachable!(),
        _ => unreachable!(),
    }
}
