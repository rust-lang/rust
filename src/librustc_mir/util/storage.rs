use rustc_index::bit_set::BitSet;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::{self, Local, Location};

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
        let mut ret = AlwaysLiveLocals(BitSet::new_filled(body.local_decls.len()));

        let mut vis = StorageAnnotationVisitor(&mut ret);
        vis.visit_body(body);

        ret
    }

    pub fn into_inner(self) -> BitSet<Local> {
        self.0
    }
}

impl std::ops::Deref for AlwaysLiveLocals {
    type Target = BitSet<Local>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Removes locals that have `Storage*` annotations from `AlwaysLiveLocals`.
struct StorageAnnotationVisitor<'a>(&'a mut AlwaysLiveLocals);

impl Visitor<'tcx> for StorageAnnotationVisitor<'_> {
    fn visit_statement(&mut self, statement: &mir::Statement<'tcx>, _location: Location) {
        use mir::StatementKind::{StorageDead, StorageLive};
        if let StorageLive(l) | StorageDead(l) = statement.kind {
            (self.0).0.remove(l);
        }
    }
}
