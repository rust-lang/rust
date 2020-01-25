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

pub(super) fn optimize_function(ctx: &mut Context, cold_ebbs: &EntitySet<Ebb>) {
    return; // FIXME add ebb arguments back

    // FIXME Move the ebb in place instead of remove and append once
    // bytecodealliance/cranelift#1339 is implemented.

    let mut ebb_insts = HashMap::new();
    for ebb in cold_ebbs.keys().filter(|&ebb| cold_ebbs.contains(ebb)) {
        let insts = ctx.func.layout.ebb_insts(ebb).collect::<Vec<_>>();
        for &inst in &insts {
            ctx.func.layout.remove_inst(inst);
        }
        ebb_insts.insert(ebb, insts);
        ctx.func.layout.remove_ebb(ebb);
    }

    // And then append them at the back again.
    for ebb in cold_ebbs.keys().filter(|&ebb| cold_ebbs.contains(ebb)) {
        ctx.func.layout.append_ebb(ebb);
        for inst in ebb_insts.remove(&ebb).unwrap() {
            ctx.func.layout.append_inst(inst, ebb);
        }
    }
}
