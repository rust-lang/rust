// skip-filecheck
#![allow(incomplete_features, unused_labels)]
#![feature(loop_match)]
#![crate_type = "lib"]

// Regression test for <https://github.com/rust-lang/rust/issues/143806>
// This used to avoid building invalid MIR with a self-assignment like `_1 = copy _1`.

// EMIT_MIR loop_match_no_self_assign.helper.built.after.mir
fn helper() -> u8 {
    let mut state = 0u8;
    #[loop_match]
    'a: loop {
        state = 'blk: {
            match state {
                _ => break 'blk state,
            }
        }
    }
}
