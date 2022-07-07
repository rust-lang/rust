use rustc_index::bit_set::BitSet;
use rustc_middle::mir::{self, HasLocalDecls, Local};

/// Returns the set of locals in a MIR body that do not have `StorageLive`/`StorageDead`
/// annotations.
///
/// These locals have fixed storage for the duration of the body.
pub fn always_live_locals(body: &mir::Body<'_>) -> BitSet<Local> {
    let mut always_live_locals = BitSet::new_filled(body.local_decls.len());

    for (local, local_decl) in body.local_decls().iter_enumerated() {
        if !local_decl.always_storage_live {
            always_live_locals.remove(local);
        }
    }

    always_live_locals
}
