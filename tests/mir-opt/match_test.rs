// skip-filecheck
// Make sure redundant testing paths in `match` expressions are sorted out.

#![feature(exclusive_range_pattern)]

// EMIT_MIR match_test.main.SimplifyCfg-initial.after.mir
fn main() {
    let x = 3;
    let b = true;

    // When `(0..=10).contains(x) && !b`, we should jump to the last arm
    // without testing two other candidates.
    match x {
        0..10 if b => 0,
        10..=20 => 1,
        -1 => 2,
        _ => 3,
    };
}
