// skip-filecheck

// EMIT_MIR or_pattern.shortcut_second_or.SimplifyCfg-initial.after.mir
fn shortcut_second_or() {
    // Check that after matching `0`, failing to match `2 | 3` skips trying to match `(1, 2 | 3)`.
    match ((0, 0), 0) {
        (x @ (0, _) | x @ (_, 1), y @ 2 | y @ 3) => {}
        _ => {}
    }
}

// EMIT_MIR or_pattern.single_switchint.SimplifyCfg-initial.after.mir
fn single_switchint() {
    // Check how many `SwitchInt`s we do. In theory a single one is necessary.
    match (1, true) {
        (1, true) => 1,
        (2, false) => 2,
        (1 | 2, true | false) => 3,
        (3 | 4, true | false) => 4,
        _ => 5,
    };
}

fn main() {}
