use rustc_index::bit_set::GrowableBitSet;
use rustc_middle::mir::{self, Local};

/// The set of locals in a MIR body that do not have `StorageLive`/`StorageDead` annotations.
///
/// These locals have fixed storage for the duration of the body.
pub fn always_storage_live_locals(body: &mir::Body<'_>, locals: &mut GrowableBitSet<Local>) {
    locals.fill(body.local_decls.len());
    let always_live_locals = locals;

    for block in &*body.basic_blocks {
        for statement in &block.statements {
            use mir::StatementKind::{StorageDead, StorageLive};
            if let StorageLive(l) | StorageDead(l) = statement.kind {
                always_live_locals.remove(l);
            }
        }
    }
}
