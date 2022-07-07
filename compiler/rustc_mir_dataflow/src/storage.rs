use rustc_index::bit_set::BitSet;
use rustc_middle::mir::{self, HasLocalDecls, Local};

/// The set of locals in a MIR body that do not have `StorageLive`/`StorageDead` annotations.
///
/// These locals have fixed storage for the duration of the body.
//
// FIXME: Currently, we need to traverse the entire MIR to compute this. We should instead store it
// as a field in the `LocalDecl` for each `Local`.
pub fn always_storage_live_locals(body: &mir::Body<'_>) -> BitSet<Local> {
    let mut always_live_locals = BitSet::new_filled(body.local_decls.len());

    for local in body.local_decls() {
        if !local.always_storage_live {
            always_live_locals.remove(local);
        }
    }

    always_live_locals
}
