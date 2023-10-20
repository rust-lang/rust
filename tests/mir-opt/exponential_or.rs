// skip-filecheck
// Test that simple or-patterns don't get expanded to exponentially large CFGs

// EMIT_MIR exponential_or.match_tuple.SimplifyCfg-initial.after.mir
fn match_tuple(x: (u32, bool, Option<i32>, u32)) -> u32 {
    match x {
        (y @ (1 | 4), true | false, Some(1 | 8) | None, z @ (6..=9 | 13..=16)) => y ^ z,
        _ => 0,
    }
}

fn main() {}
