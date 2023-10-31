// Test that MatchBranchSimplification doesn't ICE on a SwitchInt where
// one of the targets is the block that the SwitchInt terminates.
#![crate_type = "lib"]
#![feature(core_intrinsics, custom_mir)]
use std::intrinsics::mir::*;

// EMIT_MIR switch_to_self.test.MatchBranchSimplification.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn test(x: bool) {
    mir!(
        {
            Goto(bb0)
        }
        bb0 = {
            match x { false => bb0, _ => bb1 }
        }
        bb1 = {
            match x { false => bb0, _ => bb1 }
        }
    )
}
