//! This optimization moves cold code to the end of the function.
//!
//! Some code is executed much less often than other code. For example panicking or the
//! landingpads for unwinding. By moving this cold code to the end of the function the average
//! amount of jumps is reduced and the code locality is improved.
//!
//! # Undefined behaviour
//!
//! This optimization doesn't assume anything that isn't already assumed by Cranelift itself.

use crate::prelude::*;

pub(super) fn optimize_function(ctx: &mut Context, cold_blocks: &EntitySet<Block>) {
    // FIXME Move the block in place instead of remove and append once
    // bytecodealliance/cranelift#1339 is implemented.

    let mut block_insts = FxHashMap::default();
    for block in cold_blocks
        .keys()
        .filter(|&block| cold_blocks.contains(block))
    {
        let insts = ctx.func.layout.block_insts(block).collect::<Vec<_>>();
        for &inst in &insts {
            ctx.func.layout.remove_inst(inst);
        }
        block_insts.insert(block, insts);
        ctx.func.layout.remove_block(block);
    }

    // And then append them at the back again.
    for block in cold_blocks
        .keys()
        .filter(|&block| cold_blocks.contains(block))
    {
        ctx.func.layout.append_block(block);
        for inst in block_insts.remove(&block).unwrap() {
            ctx.func.layout.append_inst(inst, block);
        }
    }
}
