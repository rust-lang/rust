// skip-filecheck
// Check specific cases of sorting candidates in match lowering.
#![feature(exclusive_range_pattern)]

// EMIT_MIR sort_candidates.disjoint_ranges.SimplifyCfg-initial.after.mir
fn disjoint_ranges() {
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

fn main() {}
