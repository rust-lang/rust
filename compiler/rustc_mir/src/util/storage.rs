use rustc_index::bit_set::BitSet;
use rustc_middle::mir::{self, Local};

/// The set of locals in a MIR body that do not have `StorageLive`/`StorageDead` annotations.
///
/// These locals have fixed storage for the duration of the body.
//
// FIXME: Currently, we need to traverse the entire MIR to compute this. We should instead store it
// as a field in the `LocalDecl` for each `Local`.
#[derive(Debug, Clone)]
pub struct AlwaysLiveLocals(BitSet<Local>);

impl AlwaysLiveLocals {
    pub fn new(body: &mir::Body<'tcx>) -> Self {
        let mut always_live_locals = AlwaysLiveLocals(BitSet::new_filled(body.local_decls.len()));

        for block in body.basic_blocks() {
            for statement in &block.statements {
                use mir::StatementKind::{StorageDead, StorageLive};
                if let StorageLive(l) | StorageDead(l) = statement.kind {
                    always_live_locals.0.remove(l);
                }
            }
        }

        always_live_locals
    }

    pub fn into_inner(self) -> BitSet<Local> {
        self.0
    }
}

impl std::ops::Deref for AlwaysLiveLocals {
    type Target = BitSet<Local>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
