// skip-filecheck

// EMIT_MIR or_patterns.exponential.SimplifyCfg-initial.after.mir
fn exponential(x: (u32, bool, Option<i32>, u32)) -> u32 {
    // Test that simple or-patterns don't get expanded to exponentially large CFGs
    match x {
        (y @ (1 | 4), true | false, Some(1 | 8) | None, z @ (6..=9 | 13..=16)) => y ^ z,
        _ => 0,
    }
}

// EMIT_MIR or_patterns.simplification_subtleties.built.after.mir
fn simplification_subtleties() {
    // Test that we don't naively sort the two `2`s together and confuse the failure paths.
    match (1, true) {
        (1 | 2, false | false) => unreachable!(),
        (2, _) => unreachable!(),
        _ => {}
    }
}

fn main() {}
